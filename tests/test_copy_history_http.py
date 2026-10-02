"""工作副本版本回看：HTTP 接口端到端测试（FastAPI TestClient）。

- ``GET /api/copies/{id}/revisions`` 按 revision 升序列出版本及操作信息；
- ``GET /api/copies/{id}/revisions/{revision}`` 读取完整快照、不改变当前状态；
- 未知副本 / 未记录版本（含 0、负数、非整数）一律 404；
- 原路由、响应与特殊 id 行为不受影响。
"""

import importlib
from urllib.parse import quote

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COPY_DB_PATH", str(tmp_path / "copies.db"))
    import app.main as main
    importlib.reload(main)  # 用新环境变量重新初始化全局仓库
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
    main._store = None


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


def _create(client, text="x" * 30, anns=None):
    r = client.post("/api/copies", json={"text": text,
                                         "annotations": anns or []})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _history(client, cid):
    r = client.get(f"/api/copies/{cid}/revisions")
    assert r.status_code == 200, r.text
    return r.json()["revisions"]


def test_list_revisions_after_full_lifecycle(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7),
                                ann("c", 12, 15)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    client.patch(f"/api/copies/{cid}/annotations/a",
                 json={"reviewer": "alice", "start": 0, "end": 2,
                       "label": "甲改"})
    client.post(f"/api/copies/{cid}/annotations/a/release",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/b/release",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/finalize",
                json={"expected_revision": 5})

    revs = _history(client, cid)
    assert [r["revision"] for r in revs] == [1, 2, 3, 4, 5, 6]
    # 升序
    assert [r["revision"] for r in revs] == sorted(r["revision"] for r in revs)
    assert [(r["operation"], r["reviewer"], r["status"]) for r in revs] == [
        ("create", None, "open"),
        ("claim", "alice", "open"),
        ("modify", "alice", "open"),
        ("release", "alice", "open"),
        ("release", "alice", "open"),
        ("finalize", None, "finalized"),
    ]


def test_list_entry_is_full_snapshot_with_text_holders_groups(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7),
                                ann("c", 12, 15)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    r2 = _history(client, cid)[1]
    assert r2["text"] == "x" * 30
    assert {a["id"]: a["holder"] for a in r2["annotations"]} == \
        {"a": "alice", "b": "alice", "c": None}
    assert [g["annotation_ids"] for g in r2["groups"]] == [["a", "b"], ["c"]]
    assert [a["id"] for a in r2["annotations"]] == ["a", "b", "c"]


def test_get_single_revision_full_snapshot(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    r = client.get(f"/api/copies/{cid}/revisions/1")
    assert r.status_code == 200
    snap = r.json()
    assert snap["revision"] == 1
    assert snap["operation"] == "create" and snap["reviewer"] is None
    assert snap["status"] == "open"
    assert set(snap) >= {"id", "text", "revision", "status",
                         "annotations", "groups", "operation", "reviewer"}
    # 与列表中的同一条一致
    assert snap == _history(client, cid)[0]


def test_revision_reads_do_not_change_current_state(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    before = client.get(f"/api/copies/{cid}").json()
    client.get(f"/api/copies/{cid}/revisions")
    client.get(f"/api/copies/{cid}/revisions/1")
    client.get(f"/api/copies/{cid}/revisions/2")
    after = client.get(f"/api/copies/{cid}").json()
    assert after == before
    # 历史记录条数也不变
    assert len(_history(client, cid)) == 2


def test_idempotent_claim_and_failures_add_no_revision(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    for _ in range(2):  # 本人重复领取
        client.post(f"/api/copies/{cid}/annotations/b/claim",
                    json={"reviewer": "alice"})
    # 他人领取 409、非法 reviewer 422
    assert client.post(f"/api/copies/{cid}/annotations/a/claim",
                       json={"reviewer": "bob"}).status_code == 409
    assert client.post(f"/api/copies/{cid}/annotations/a/claim",
                       json={"reviewer": ""}).status_code == 422
    revs = _history(client, cid)
    assert [(r["revision"], r["operation"]) for r in revs] == \
        [(1, "create"), (2, "claim")]


def test_list_revisions_unknown_copy_404(client):
    assert client.get("/api/copies/deadbeef/revisions").status_code == 404


def test_get_revision_unknown_copy_404(client):
    assert client.get("/api/copies/deadbeef/revisions/1").status_code == 404


def test_get_unrecorded_revision_404(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    assert client.get(f"/api/copies/{cid}/revisions/2").status_code == 404
    assert client.get(f"/api/copies/{cid}/revisions/99").status_code == 404


@pytest.mark.parametrize("bad", ["0", "-1", "-3", "abc", "1.5", "true"])
def test_invalid_revision_path_404(client, bad):
    cid = _create(client, anns=[ann("a", 0, 3)])
    assert client.get(f"/api/copies/{cid}/revisions/{bad}").status_code == 404


def test_special_ids_history_recorded_and_original_order_kept(client):
    # TestClient 对路径百分号解码两次，无法区分 "/" 与字面 "%2F"（该区分由
    # test_copy_special_ids_http 的真实 uvicorn 用例覆盖）；这里用空串、
    # 含 "/"、含换行的 id 验证历史留痕与**原排序**保留即可。
    cid = _create(client, anns=[
        ann("", 0, 2, "空"),
        ann("审阅/1", 5, 8, "斜"),
        ann("a\n", 20, 22, "换行"),
    ])
    assert client.post(f"/api/copies/{cid}/annotations//claim",
                       json={"reviewer": "alice"}).status_code == 200
    assert client.patch(f"/api/copies/{cid}/annotations/",
                        json={"reviewer": "alice", "start": 0, "end": 1,
                              "label": "空改"}).status_code == 200
    assert client.post(
        f"/api/copies/{cid}/annotations/{quote('审阅/1', safe='')}/claim",
        json={"reviewer": "bob"}).status_code == 200
    assert client.post(f"/api/copies/{cid}/annotations/a%0A/claim",
                       json={"reviewer": "dave"}).status_code == 200

    revs = _history(client, cid)
    assert [(r["operation"], r["reviewer"]) for r in revs] == [
        ("create", None),
        ("claim", "alice"),
        ("modify", "alice"),
        ("claim", "bob"),
        ("claim", "dave"),
    ]
    # 特殊 id 的**原排序**原样进入每条历史快照
    for r in revs:
        assert [a["id"] for a in r["annotations"]] == ["", "审阅/1", "a\n"]
    # 单版本读取同样保留特殊 id 与该版本持有者
    r5 = client.get(f"/api/copies/{cid}/revisions/5").json()
    assert {a["id"]: a["holder"] for a in r5["annotations"]} == \
        {"": "alice", "审阅/1": "bob", "a\n": "dave"}
    # 原 GET 快照路由行为不变
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["revision"] == 5 and snap["status"] == "open"
