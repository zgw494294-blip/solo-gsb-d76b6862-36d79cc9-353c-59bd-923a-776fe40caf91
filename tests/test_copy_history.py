"""工作副本版本回看（revision history）：SQLite 仓库测试。

覆盖：创建即保存 revision=1 快照；领取 / 释放 / 修改 / 结稿递增时保存该版本
文本、批注（原排序）、持有者、status 及操作类型 / reviewer；重复领取与失败
请求不产生记录；快照与副本变更同事务原子提交；旧库仅补当前 revision 的
baseline，不编造早期历史，重复启动不重复补记；历史跨重启保留。
"""

import sqlite3
import threading

import pytest

from app.copies import (
    CopyNotFound,
    CopyStore,
    GroupHeldError,
    RevisionNotFound,
)

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def _ops(store, cid):
    return [(r["revision"], r["operation"], r["reviewer"], r["status"])
            for r in store.list_revisions(cid)]


# ---------- 创建即 revision=1 ----------

def test_create_persists_revision_one_snapshot(store):
    # 批注故意不按稳定顺序传入：历史快照应保留**原排序**
    cid = store.create(TEXT, [ann("b", 3, 7, "乙"), ann("a", 0, 4, "甲"),
                              ann("c", 12, 15, "丙")])["id"]
    revs = store.list_revisions(cid)
    assert len(revs) == 1
    r1 = revs[0]
    assert r1["revision"] == 1 and r1["status"] == "open"
    assert r1["operation"] == "create" and r1["reviewer"] is None
    assert r1["text"] == TEXT
    assert [a["id"] for a in r1["annotations"]] == ["b", "a", "c"]
    assert all(a["holder"] is None for a in r1["annotations"])
    # 连通组按该版本几何重算
    assert [g["annotation_ids"] for g in r1["groups"]] == [["a", "b"], ["c"]]


def test_get_revision_one_equals_list_entry(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    got = store.get_revision(cid, 1)
    assert got == store.list_revisions(cid)[0]


# ---------- 递增操作逐个留痕 ----------

def test_each_mutating_operation_records_snapshot(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    store.claim(cid, "a", "alice")            # 2 claim
    store.modify(cid, "a", "alice", 0, 2, "甲改")  # 3 modify（拆开同组）
    store.release(cid, "b", "alice")          # 4 release（b 碎片）
    store.release(cid, "a", "alice")          # 5 release
    store.finalize(cid, 5)                    # 6 finalize

    assert _ops(store, cid) == [
        (1, "create", None, "open"),
        (2, "claim", "alice", "open"),
        (3, "modify", "alice", "open"),
        (4, "release", "alice", "open"),
        (5, "release", "alice", "open"),
        (6, "finalize", None, "finalized"),
    ]


def test_revision_snapshots_capture_state_at_each_version(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    store.claim(cid, "a", "alice")
    store.claim(cid, "c", "bob")

    r2 = store.get_revision(cid, 2)
    assert {a["id"]: a["holder"] for a in r2["annotations"]} == \
        {"a": "alice", "b": "alice", "c": None}
    r3 = store.get_revision(cid, 3)
    assert {a["id"]: a["holder"] for a in r3["annotations"]} == \
        {"a": "alice", "b": "alice", "c": "bob"}

    # 修改后：a 缩短不再与 b 正长度重叠 → 三组；a 仍 alice、b 保留 alice
    store.modify(cid, "a", "alice", 0, 2, "X")
    r4 = store.get_revision(cid, 4)
    by_id = {a["id"]: a for a in r4["annotations"]}
    assert (by_id["a"]["start"], by_id["a"]["end"],
            by_id["a"]["label"]) == (0, 2, "X")
    assert {a["id"]: a["holder"] for a in r4["annotations"]} == \
        {"a": "alice", "b": "alice", "c": "bob"}
    assert [g["annotation_ids"] for g in r4["groups"]] == \
        [["a"], ["b"], ["c"]]

    # 早期版本几何不变（不可变历史）
    old = store.get_revision(cid, 1)
    by_id = {a["id"]: a for a in old["annotations"]}
    assert (by_id["a"]["start"], by_id["a"]["end"],
            by_id["a"]["label"]) == (0, 4, "L")

    store.release(cid, "a", "alice")
    store.release(cid, "b", "alice")
    store.release(cid, "c", "bob")
    store.finalize(cid, 7)
    assert store.get_revision(cid, 8)["status"] == "finalized"
    # 结稿快照文本、批注、持有者与排序均不变
    r7 = store.get_revision(cid, 7)
    r8 = store.get_revision(cid, 8)
    assert r8["text"] == r7["text"]
    assert r8["annotations"] == r7["annotations"]
    assert [a["id"] for a in r8["annotations"]] == ["a", "b", "c"]


def test_reviewer_is_recorded_per_operation(store):
    cid = store.create(TEXT, [ann("a", 0, 3), ann("b", 6, 9)])["id"]
    store.claim(cid, "a", "alice")
    store.modify(cid, "a", "alice", 0, 2, "X")
    store.claim(cid, "b", "bob")
    store.release(cid, "b", "bob")
    reviewers = {r["revision"]: r["reviewer"]
                 for r in store.list_revisions(cid)}
    assert reviewers == {1: None, 2: "alice", 3: "alice",
                         4: "bob", 5: "bob"}


# ---------- 重复领取与失败请求不产生记录 ----------

def test_idempotent_repeated_claim_creates_no_revision(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    assert store.claim(cid, "a", "alice")["revision"] == 2
    for _ in range(3):  # 本人重复领取幂等
        snap = store.claim(cid, "b", "alice")
        assert snap["revision"] == 2
    assert _ops(store, cid) == [(1, "create", None, "open"),
                                (2, "claim", "alice", "open")]


def test_failed_requests_create_no_revision(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")  # revision 2
    from app.copies import (
        CopyStillHeldError,
        CopyValidationError,
        NotHeldError,
        RevisionMismatchError,
    )
    # 他人领取 → 409
    with pytest.raises(GroupHeldError):
        store.claim(cid, "b", "bob")
    # 非持有者修改 → 409
    with pytest.raises(NotHeldError):
        store.modify(cid, "a", "bob", 0, 1, "X")
    # 结稿版本过期 / 仍有持有者 → 409
    with pytest.raises(RevisionMismatchError):
        store.finalize(cid, 1)
    with pytest.raises(CopyStillHeldError):
        store.finalize(cid, 2)
    # 非法 reviewer → 422
    with pytest.raises(CopyValidationError):
        store.claim(cid, "a", "")

    snap = store.get(cid)
    assert snap["revision"] == 2
    assert _ops(store, cid) == [(1, "create", None, "open"),
                                (2, "claim", "alice", "open")]


# ---------- 原子性：快照写失败 → 副本变更回滚 ----------

def test_revision_insert_failure_rolls_back_copy_change(tmp_path):
    store = CopyStore(tmp_path / "copies.db")
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]

    def boom(conn, state, operation, reviewer):
        raise RuntimeError("simulated snapshot write failure")

    store._insert_revision = staticmethod(boom)
    with pytest.raises(RuntimeError):
        store.claim(cid, "a", "alice")
    del store._insert_revision  # 恢复类上的静态方法

    snap = store.get(cid)
    assert snap["revision"] == 1
    assert snap["annotations"][0]["holder"] is None
    assert _ops(store, cid) == [(1, "create", None, "open")]


# ---------- 并发：快照片版本不混入后续版本 ----------

def test_concurrent_modifications_record_every_version_coherently(store):
    anns = [ann(f"a{i}", 3 * i, 3 * i + 2) for i in range(10)]
    cid = store.create(TEXT, anns)["id"]
    for i in range(10):
        store.claim(cid, f"a{i}", f"r{i}")

    barrier = threading.Barrier(10)

    def edit(i):
        barrier.wait()
        store.modify(cid, f"a{i}", f"r{i}", 3 * i, 3 * i + 1, f"lab{i}")

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    revs = store.list_revisions(cid)
    # 1（create）+ 10（claim）+ 10（modify），revision 连续、升序、无缺失
    assert [r["revision"] for r in revs] == list(range(1, 22))
    # 每条记录自洽：revision 字段、该时刻已生效修改的几何单调累积
    edited: set[str] = set()
    for r in revs[11:]:
        now = {a["id"] for a in r["annotations"]
               if a["label"].startswith("lab")}
        assert edited < now and len(now) == len(edited) + 1
        edited = now
        # 记录内文本与几何属于该版本，不可能是更晚版本
        assert r["text"] == TEXT
        for a in r["annotations"]:
            if a["id"] in now:
                j = int(a["id"][1:])
                assert (a["start"], a["end"]) == (3 * j, 3 * j + 1)
            else:
                j = int(a["id"][1:])
                assert (a["start"], a["end"]) == (3 * j, 3 * j + 2)
    # 最新版本记录与当前快照一致
    latest = store.get(cid)
    last = store.get_revision(cid, 21)
    assert latest["revision"] == 21
    assert {a["id"]: (a["start"], a["end"], a["label"], a["holder"])
            for a in latest["annotations"]} == \
        {a["id"]: (a["start"], a["end"], a["label"], a["holder"])
         for a in last["annotations"]}


# ---------- 404：未知副本 / 未记录版本 ----------

def test_list_revisions_unknown_copy_404(store):
    with pytest.raises(CopyNotFound):
        store.list_revisions("deadbeef")


def test_get_revision_unknown_copy_404(store):
    with pytest.raises(CopyNotFound):
        store.get_revision("deadbeef", 1)


def test_get_unrecorded_revision_404(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")
    with pytest.raises(RevisionNotFound):
        store.get_revision(cid, 99)


def test_history_reads_do_not_change_state(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")
    before = store.get(cid)
    store.list_revisions(cid)
    store.get_revision(cid, 1)
    store.get_revision(cid, 2)
    assert store.get(cid) == before


# ---------- 跨重启保留 ----------

def test_history_retained_across_reopen(tmp_path):
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    s1.claim(cid, "a", "alice")
    s1.modify(cid, "a", "alice", 0, 2, "新")

    s2 = CopyStore(db)  # 模拟容器重启
    revs = s2.list_revisions(cid)
    assert [(r["revision"], r["operation"]) for r in revs] == \
        [(1, "create"), (2, "claim"), (3, "modify")]
    assert s2.get_revision(cid, 2)["annotations"][0]["holder"] == "alice"
    # 重启后继续追加新版本
    s2.release(cid, "b", "alice")
    assert [(r["revision"], r["operation"])
            for r in CopyStore(db).list_revisions(cid)] == \
        [(1, "create"), (2, "claim"), (3, "modify"), (4, "release")]


# ---------- 旧库迁移：只补当前 revision 的 baseline ----------

def _build_old_db(db, revision=5, status_less=True):
    conn = sqlite3.connect(db)
    if status_less:
        conn.executescript("""
            CREATE TABLE copies (
                id TEXT PRIMARY KEY, text TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE annotations (
                copy_id TEXT NOT NULL, ann_id TEXT NOT NULL, pos INTEGER NOT NULL,
                start INTEGER NOT NULL, end INTEGER NOT NULL, label TEXT NOT NULL,
                holder TEXT, PRIMARY KEY (copy_id, ann_id));
        """)
    conn.execute("INSERT INTO copies(id, text, revision) VALUES('old1', ?, ?)",
                 (TEXT, revision))
    conn.execute("INSERT INTO annotations VALUES('old1','a',0,0,4,'L','alice')")
    conn.execute("INSERT INTO annotations VALUES('old1','b',1,3,7,'M','alice')")
    conn.commit()
    conn.close()


def test_old_db_backfills_only_current_revision_baseline(tmp_path):
    db = tmp_path / "copies.db"
    _build_old_db(db, revision=5)

    store = CopyStore(db)
    revs = store.list_revisions("old1")
    assert [(r["revision"], r["operation"], r["reviewer"]) for r in revs] == \
        [(5, "baseline", None)]
    baseline = revs[0]
    assert baseline["status"] == "open" and baseline["text"] == TEXT
    # 原排序、持有者、几何整体保留
    assert [(a["id"], a["holder"]) for a in baseline["annotations"]] == \
        [("a", "alice"), ("b", "alice")]
    assert [g["annotation_ids"] for g in baseline["groups"]] == [["a", "b"]]
    # 不编造早期历史：1~4 均未记录 → 404
    for missing in (1, 2, 3, 4):
        with pytest.raises(RevisionNotFound):
            store.get_revision("old1", missing)
    # 原数据不改
    assert store.get("old1")["revision"] == 5


def test_old_db_baseline_then_new_versions_append(tmp_path):
    db = tmp_path / "copies.db"
    _build_old_db(db, revision=5)
    s1 = CopyStore(db)
    s1.release("old1", "a", "alice")  # revision 6

    s2 = CopyStore(db)  # 重启：baseline 不重复补记，release 记录保留
    assert [(r["revision"], r["operation"])
            for r in s2.list_revisions("old1")] == \
        [(5, "baseline"), (6, "release")]
    CopyStore(db)
    CopyStore(db)  # 多次启动幂等
    assert [(r["revision"], r["operation"])
            for r in CopyStore(db).list_revisions("old1")] == \
        [(5, "baseline"), (6, "release")]


def test_backfill_does_not_duplicate_for_newly_created_copies(tmp_path):
    # 新库创建的副本本身就有 create 记录，迁移逻辑不得再补 baseline
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 3)])["id"]
    s2 = CopyStore(db)  # 重启
    assert [(r["revision"], r["operation"])
            for r in s2.list_revisions(cid)] == [(1, "create")]


def test_old_db_without_status_column_and_history_together(tmp_path):
    # 补 status 列与补 baseline 在同一次启动完成，且后续释放 / 结稿正常留痕
    db = tmp_path / "copies.db"
    _build_old_db(db, revision=1, status_less=True)
    store = CopyStore(db)
    assert store.get("old1")["status"] == "open"
    store.release("old1", "a", "alice")  # revision 2：整组释放
    done = store.finalize("old1", 2)
    assert done["revision"] == 3 and done["status"] == "finalized"
    assert [(r["revision"], r["operation"], r["status"])
            for r in store.list_revisions("old1")] == [
        (1, "baseline", "open"), (2, "release", "open"),
        (3, "finalize", "finalized")]
