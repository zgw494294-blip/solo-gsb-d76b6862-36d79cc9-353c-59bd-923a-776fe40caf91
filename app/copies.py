"""可领取的审阅工作副本核心逻辑与 SQLite 持久化。

偏移约定：所有 start/end 均为 Unicode 码点偏移，区间为左闭右开 [start, end)。

一个工作副本包含**固定文本**与若干批注（唯一字符串 id、start、end、非空
label），初始**无人持有**。

- **分组**：两条批注正长度重叠（区间交集长度 > 0，即 ``a.start < b.end`` 且
  ``b.start < a.end``）即在二者间连边，传递闭包得到连通组；**相接**
  （``end == start``）不连边。
- **持有不变量**：同一连通组的全部批注要么无人持有，要么由同一非空审阅者
  标识持有。领取 / 释放均按批注 id 定位其所在组并**整组**操作。
- **领取**：组无人持有 → 整组归该审阅者；已由本人持有 → 幂等成功且不产生
  新版本；由他人持有 → `GroupHledError`（409）。
- **释放**：仅持有者本人可释放整组；无人持有 → `NotHeldError`，他人持有 →
  `GroupHeldError`（均 409）。
- **修改**：仅持有者本人可修改单条批注的范围与 label，不增删批注。修改后
  在新几何上重算分组：新并入的未领取批注归该持有者；新组触及其他持有者的
  批注 → `ModifyConflictError`（409，范围与持有权均不变）；拆开的组保留原
  持有者。
- **结稿**：携带 `expected_revision`（正整数）把副本从 `open` 结稿为
  `finalized`——不可再改的定稿，不提供重开。已结稿 → `CopyFinalizedError`、
  版本过期 → `RevisionMismatchError`、仍有持有者 → `CopyStillHeldError`
  （均 409，失败不改任何状态）。成功仅把 `status` 改为 `finalized` 并
  `revision += 1`，文本、批注、持有权与排序不变。结稿后领取 / 释放 / 修改
  一律 `CopyFinalizedError`（409）；结稿与其他变更在锁内串行，等价于某个
  先后次序，不会出现结稿成功后仍写入修改或领取的情况。

`CopyState` 为不依赖数据库的纯内存核心（便于单测）；`CopyStore` 用 SQLite
持久化，每个变更操作在进程级锁内「完整读入 → 纯核心运算 → 整体写回」，
因此并发请求等价于逐次完整执行，不会丢失其他批注的修改。旧库启动时自动
补 `status` 列，旧副本默认 `open` 且原 `revision` 不变。

**版本回看**：每次成功提交都在同一事务内写入一条不可变版本记录——创建时
保存 `revision=1` 的完整快照（`operation=create`、`reviewer=null`）；领取 /
释放 / 修改 / 结稿使 `revision` 递增时保存新版本的文本、批注（**原排序**）、
持有者与 `status`，并附操作类型（`claim` / `release` / `modify` /
`finalize`）与 `reviewer`（结稿为 `null`）。本人重复领取（幂等、不增版本）
及一切失败请求（422 / 409 / 404）**不产生记录**；快照与副本变更同事务提交，
写入失败二者均不变，因此并发时已提交的快照永远对应该次提交、不会混入后续
版本。旧库（无版本表的数据库）启动时只为每个旧副本补**当前 revision** 的一条
`operation=baseline` 快照，不编造早期历史、不改原数据，重复启动不重复补记；
历史随原卷在同一数据库中跨重启保留。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

# ---------- 错误（HTTP 映射见 app.main） ----------


class CopyError(ValueError):
    """工作副本业务错误基类（409 / 422）。"""


class CopyValidationError(CopyError):
    """非法输入（对应 HTTP 422）。"""


class GroupHeldError(CopyError):
    """目标组正由其他审阅者持有（对应 HTTP 409）。"""

    def __init__(self, holder: str, group: list[str] | None = None):
        self.holder = holder
        self.group = group or []
        super().__init__(
            f"该批注所在连通组已由审阅者 {holder!r} 持有："
            + "、".join(self.group))


class NotHeldError(CopyError):
    """目标组无人持有 / 批注未由该审阅者持有（对应 HTTP 409）。"""

    def __init__(self, group: list[str] | None = None,
                 message: str = "该批注所在连通组当前无人持有，不能执行此操作"):
        self.group = group or []
        super().__init__(message)


class ModifyConflictError(CopyError):
    """修改后的新组触及其他审阅者持有的批注（对应 HTTP 409）。"""

    def __init__(self, foreign: dict[str, str]):
        # foreign: {触及到的他人批注 id: 持有者}
        self.foreign = dict(foreign)
        super().__init__(
            "修改后的连通组触及其他审阅者持有的批注："
            + "、".join(f"{i}（{h}）" for i, h in self.foreign.items()))


class CopyFinalizedError(CopyError):
    """工作副本已结稿，不可再变更（对应 HTTP 409）。"""

    def __init__(self, copy_id: str):
        self.copy_id = copy_id
        super().__init__(f"工作副本已结稿，不可再变更：{copy_id}")


class RevisionMismatchError(CopyError):
    """expected_revision 与当前版本不一致，版本已过期（对应 HTTP 409）。"""

    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(
            f"版本已过期：expected_revision={expected}，"
            f"当前 revision={current}")


class CopyStillHeldError(CopyError):
    """副本仍有批注被持有，不能结稿（对应 HTTP 409）。"""

    def __init__(self, holders: dict[str, str]):
        # holders: {仍被持有的批注 id: 持有者}
        self.holders = dict(holders)
        super().__init__(
            "副本仍有批注被持有，不能结稿："
            + "、".join(f"{i}（{h}）" for i, h in self.holders.items()))


class CopyNotFound(LookupError):
    """工作副本不存在（对应 HTTP 404）。"""

    def __init__(self, copy_id: str):
        self.copy_id = copy_id
        super().__init__(f"工作副本不存在：{copy_id}")


class RevisionNotFound(LookupError):
    """副本存在但该 revision 未记录（对应 HTTP 404）。"""

    def __init__(self, copy_id: str, revision: int):
        self.copy_id = copy_id
        self.revision = revision
        super().__init__(
            f"工作副本 {copy_id} 不存在已记录的 revision={revision}")


class AnnotationNotFound(LookupError):
    """批注在该工作副本中不存在（对应 HTTP 404）。"""

    def __init__(self, ann_id: str):
        self.ann_id = ann_id
        super().__init__(f"批注不存在：{ann_id}")


# ---------- 输入校验 ----------


def check_reviewer(reviewer) -> None:
    """审阅者标识必须是非空字符串。"""
    if not isinstance(reviewer, str) or not reviewer:
        raise CopyValidationError("reviewer 必须是非空字符串")


def check_label(label) -> None:
    """label 必须是非空字符串。"""
    if not isinstance(label, str) or not label:
        raise CopyValidationError("label 必须是非空字符串")


def check_interval(start, end, text_len: int) -> None:
    """码点区间 [start, end) 必须满足 0 ≤ start < end ≤ 文本长度。"""
    if isinstance(start, bool) or isinstance(end, bool) or \
            not isinstance(start, int) or not isinstance(end, int) \
            or not 0 <= start < end <= text_len:
        raise CopyValidationError(
            f"区间 [{start}, {end}) 非法或超出文本长度 {text_len}"
            "（须满足 0 ≤ start < end ≤ 文本长度）")


def check_expected_revision(value) -> None:
    """expected_revision 必须是正整数（布尔不是合法版本号）。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CopyValidationError(
            f"expected_revision 必须是正整数：{value!r}")


# ---------- 分组（并查集） ----------


def _sort_key(a: dict) -> tuple:
    """批注稳定排序键：start、end、label、id。"""
    return (a["start"], a["end"], a["label"], str(a["id"]))


def connected_groups(annotations: list[dict]) -> list[list[str]]:
    """计算正长度重叠的传递连通组。

    相接（``end == start``）不连边。返回组的列表：组按组内最小排序键
    （即最早成员）排序，组内批注 id 按 (start, end, label, id) 排序。
    """
    ordered = sorted(annotations, key=_sort_key)
    n = len(ordered)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        # start 已升序：ordered[j].start >= ordered[i].end 时，j 及其后
        # 都不可能与 i 正长度重叠（相接不算），可提前结束。
        for j in range(i + 1, n):
            if ordered[j]["start"] >= ordered[i]["end"]:
                break
            union(i, j)

    groups: dict[int, list[str]] = {}
    for i, a in enumerate(ordered):
        groups.setdefault(find(i), []).append(a["id"])
    # find 根取最小下标，即组内最早成员，按首次出现顺序输出天然有序
    return [groups[k] for k in sorted(groups)]


# ---------- 纯内存核心 ----------


class CopyState:
    """单个工作副本的内存状态（持有不变量见模块 docstring）。"""

    def __init__(self, copy_id: str, text: str,
                 annotations: list[dict], revision: int = 1,
                 status: str = "open"):
        self.copy_id = copy_id
        self.text = text
        self.order: list[str] = [a["id"] for a in annotations]
        self.annotations: dict[str, dict] = {
            a["id"]: {"id": a["id"], "start": a["start"],
                      "end": a["end"], "label": a["label"]}
            for a in annotations
        }
        self.holders: dict[str, str] = {}
        self.revision = revision
        self.status = status  # "open" | "finalized"（结稿后不可再变更）

    def _check_open(self) -> None:
        """已结稿的副本不可再变更：领取 / 释放 / 修改 / 再次结稿均 409。"""
        if self.status == "finalized":
            raise CopyFinalizedError(self.copy_id)

    def _all_annotations(self) -> list[dict]:
        return [self.annotations[i] for i in self.order]

    def group_of(self, ann_id: str) -> list[str]:
        """返回批注所在连通组的全部 id（按稳定顺序）；批注不存在则报错。"""
        if ann_id not in self.annotations:
            raise AnnotationNotFound(ann_id)
        for group in connected_groups(self._all_annotations()):
            if ann_id in group:
                return group
        raise AssertionError("连通组计算异常")  # 不可达

    def claim(self, ann_id: str, reviewer: str) -> bool:
        """领取批注所在整组。返回是否发生了状态变更（本人重复领取为 False）。"""
        check_reviewer(reviewer)
        group = self.group_of(ann_id)
        self._check_open()  # 422 / 404 之后：结稿副本上的有效领取一律 409
        owners = {self.holders[i] for i in group if self.holders.get(i) is not None}
        other = next(iter(owners - {reviewer}), None)
        if other is not None:
            raise GroupHeldError(other, group)
        changed = any(self.holders.get(i) is None for i in group)
        if changed:
            for i in group:
                self.holders[i] = reviewer
            self.revision += 1
        return changed

    def release(self, ann_id: str, reviewer: str) -> None:
        """释放批注所在整组（仅限持有者本人）。"""
        check_reviewer(reviewer)
        group = self.group_of(ann_id)
        self._check_open()
        owner = self.holders.get(group[0])  # 持有不变量：整组持有者相同
        if owner is None:
            raise NotHeldError(group)
        if owner != reviewer:
            raise GroupHeldError(owner, group)
        for i in group:
            self.holders.pop(i, None)
        self.revision += 1

    def modify(self, ann_id: str, reviewer: str,
               start: int, end: int, label: str) -> None:
        """修改单条批注的范围与 label（仅限持有者本人），随后重算分组。"""
        check_reviewer(reviewer)
        check_label(label)
        if ann_id not in self.annotations:
            raise AnnotationNotFound(ann_id)
        check_interval(start, end, len(self.text))  # 非法输入优先于持有冲突
        self._check_open()  # 422 / 404 之后：结稿副本上的有效修改一律 409
        if self.holders.get(ann_id) != reviewer:
            raise NotHeldError(
                message=f"批注 {ann_id} 未由审阅者 {reviewer!r} 持有，不能修改")
        # 在候选新几何上重算分组（不改本对象，冲突时整体放弃）
        candidate = [dict(self.annotations[i]) for i in self.order]
        for a in candidate:
            if a["id"] == ann_id:
                a["start"], a["end"], a["label"] = start, end, label
        target_ids = next(g for g in connected_groups(candidate) if ann_id in g)
        foreign = {
            i: self.holders[i]
            for i in target_ids
            if self.holders.get(i) is not None and self.holders[i] != reviewer
        }
        if foreign:
            raise ModifyConflictError(foreign)
        # 应用新范围 / label；新组成员整组归本人（含并入的未领取批注）。
        # 不在新组内的其他批注持有权一律不动——原同组被拆开的碎片因此
        # 保留原持有者（即本人），其他组完全不受影响。
        self.annotations[ann_id].update(start=start, end=end, label=label)
        for i in target_ids:
            self.holders[i] = reviewer
        self.revision += 1

    def finalize(self, expected_revision: int) -> None:
        """结稿为不可再改的定稿（不提供重开）。

        仅当副本未结稿、``expected_revision`` 与当前版本一致且无任何持有者
        时成功：仅把 ``status`` 从 ``open`` 改为 ``finalized`` 并
        ``revision += 1``，文本、批注、持有权与排序均不变；任一条件不满足
        抛错（409），状态完全不变。
        """
        check_expected_revision(expected_revision)
        self._check_open()
        if expected_revision != self.revision:
            raise RevisionMismatchError(expected_revision, self.revision)
        held = {i: h for i, h in self.holders.items() if h is not None}
        if held:
            raise CopyStillHeldError(held)
        self.status = "finalized"
        self.revision += 1

    def snapshot(self) -> dict:
        """返回同一提交状态下的文本、批注（含持有者）与连通组。"""
        anns = self._all_annotations()
        groups = connected_groups(anns)
        return {
            "id": self.copy_id,
            "text": self.text,
            "revision": self.revision,
            "status": self.status,
            "annotations": [
                {"id": a["id"], "start": a["start"], "end": a["end"],
                 "label": a["label"], "holder": self.holders.get(a["id"])}
                for a in sorted(anns, key=_sort_key)
            ],
            "groups": [
                {"annotation_ids": ids,
                 "holder": self.holders.get(ids[0]) if ids else None}
                for ids in groups
            ],
        }

    def revision_snapshot(self, operation: str,
                          reviewer: str | None) -> dict:
        """生成历史版本完整快照：文本、批注（**原排序**、含持有者）、status。

        与 :meth:`snapshot` 不同，批注按创建时的**原排序**（``self.order``）
        原样保存、不重排，因此回看任意版本时文本、批注、持有者、status 与
        原排序都能整体还原；连通组在读取时由几何重算，不入库。``operation``
        为产生该版本的操作类型（``create`` / ``claim`` / ``release`` /
        ``modify`` / ``finalize`` / ``baseline``），``reviewer`` 为操作审阅者
        （创建、结稿、基线为 ``None``）。
        """
        return {
            "id": self.copy_id,
            "text": self.text,
            "revision": self.revision,
            "status": self.status,
            "operation": operation,
            "reviewer": reviewer,
            "annotations": [
                {"id": i,
                 "start": self.annotations[i]["start"],
                 "end": self.annotations[i]["end"],
                 "label": self.annotations[i]["label"],
                 "holder": self.holders.get(i)}
                for i in self.order
            ],
        }


# ---------- SQLite 持久化 ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS copies (
    id         TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    revision   INTEGER NOT NULL DEFAULT 1,
    status     TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS annotations (
    copy_id TEXT NOT NULL REFERENCES copies(id) ON DELETE CASCADE,
    ann_id  TEXT NOT NULL,
    pos     INTEGER NOT NULL,
    start   INTEGER NOT NULL,
    end     INTEGER NOT NULL,
    label   TEXT NOT NULL,
    holder  TEXT,
    PRIMARY KEY (copy_id, ann_id)
);
CREATE INDEX IF NOT EXISTS idx_annotations_copy_pos
    ON annotations(copy_id, pos);
CREATE TABLE IF NOT EXISTS copy_revisions (
    copy_id   TEXT NOT NULL REFERENCES copies(id) ON DELETE CASCADE,
    revision  INTEGER NOT NULL,
    operation TEXT NOT NULL,
    reviewer  TEXT,
    payload   TEXT NOT NULL,
    PRIMARY KEY (copy_id, revision)
);
"""


def default_db_path() -> Path:
    """副本数据库路径：环境变量 COPY_DB_PATH 优先。

    容器内默认 ``/data/copies.db``（由 Compose 挂卷）；本地开发无 /data
    写权限时回退到仓库根目录下的 ``data/copies.db``。
    """
    env = os.environ.get("COPY_DB_PATH")
    if env:
        return Path(env)
    candidates = (
        Path("/data/copies.db"),
        Path(__file__).resolve().parent.parent / "data" / "copies.db",
    )
    for cand in candidates:
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            probe = cand.parent / ".copies-db-write-test"
            probe.write_bytes(b"ok")
            probe.unlink()
            return cand
        except OSError:
            continue
    raise RuntimeError("无法确定可写的 SQLite 数据库路径（可设置 COPY_DB_PATH）")


class CopyStore:
    """SQLite 工作副本仓库；进程级锁保证变更操作串行完整执行。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_db_path()
        self._lock = threading.RLock()
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_schema(self) -> None:
        """建表（幂等）：随容器启动初始化，重启后数据保留。

        旧库缺少 ``status`` 列时自动补列：既有副本默认 ``open``，
        原 ``revision`` 不变。

        旧库（``copy_revisions`` 版本表为空 / 新建前已存在的副本没有任何
        版本记录）启动时为每个旧副本补**当前 revision** 的一条 ``baseline``
        快照——只补起点、不编造早期历史，原副本数据（文本、批注、持有者、
        revision、status）一概不改；以「该副本尚无任何版本记录」为补记条件，
        重复启动不会重复补记。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(copies)")}
            if "status" not in cols:
                conn.execute(
                    "ALTER TABLE copies "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'open'")
            # 旧副本补当前 revision 的 baseline 快照（与补列同事务提交）：
            # NOT EXISTS 保证只在该副本一条版本记录都没有时补记，重复启动
            # 不重复补；此后副本变更产生的新版本照常追加，不影响旧副本。
            rows = conn.execute(
                "SELECT c.id FROM copies c WHERE NOT EXISTS ( "
                "SELECT 1 FROM copy_revisions r WHERE r.copy_id = c.id)"
            ).fetchall()
            for (copy_id,) in rows:
                state = self._load(conn, copy_id)
                self._insert_revision(conn, state, "baseline", None)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _load(self, conn: sqlite3.Connection, copy_id: str) -> CopyState:
        row = conn.execute(
            "SELECT text, revision, status FROM copies WHERE id=?",
            (copy_id,)).fetchone()
        if row is None:
            raise CopyNotFound(copy_id)
        text, revision, status = row
        rows = conn.execute(
            "SELECT ann_id, pos, start, end, label, holder FROM annotations "
            "WHERE copy_id=? ORDER BY pos", (copy_id,)).fetchall()
        state = CopyState(
            copy_id, text,
            [{"id": r[0], "start": r[2], "end": r[3], "label": r[4]} for r in rows],
            revision=revision, status=status)
        for r in rows:
            if r[5] is not None:
                state.holders[r[0]] = r[5]
        return state

    @staticmethod
    def _insert_revision(conn: sqlite3.Connection, state: CopyState,
                         operation: str, reviewer: str | None) -> None:
        """把该次提交的完整快照写入版本表（须由调用方在同一事务内提交）。

        快照与副本变更共享同一事务：任一步失败整体回滚，副本与版本记录
        要么同时生效、要么都不变；锁内串行使记录的 revision 必为该次提交
        的版本，不会混入并发的后续版本。
        """
        record = state.revision_snapshot(operation, reviewer)
        conn.execute(
            "INSERT INTO copy_revisions(copy_id, revision, operation, "
            "reviewer, payload) VALUES(?, ?, ?, ?, ?)",
            (state.copy_id, state.revision, operation, reviewer,
             json.dumps(record, ensure_ascii=False)))

    @staticmethod
    def _revision_view(record: dict) -> dict:
        """把存储的版本记录还原为完整快照（批注原排序 + 重算的连通组）。

        组不在版本记录中存储，读取时按该版本快照的几何与持有者重算，
        与 :meth:`CopyState.snapshot` 的结果保持同一形态。
        """
        state = CopyState(
            record["id"], record["text"], record["annotations"],
            revision=record["revision"], status=record["status"])
        for a in record["annotations"]:
            if a["holder"] is not None:
                state.holders[a["id"]] = a["holder"]
        view = state.snapshot()
        # snapshot() 的批注按稳定顺序重排；版本回看按保存时的**原排序**返回
        view["annotations"] = record["annotations"]
        view["operation"] = record["operation"]
        view["reviewer"] = record["reviewer"]
        return view

    def create(self, text: str, annotations: list[dict]) -> dict:
        """创建工作副本，返回其初始快照（初始无人持有，revision=1）。

        副本与 ``revision=1`` 的 ``create`` 版本快照在同一事务内提交。
        """
        copy_id = uuid.uuid4().hex
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO copies(id, text, revision, status) "
                    "VALUES(?, ?, 1, 'open')",
                    (copy_id, text))
                conn.executemany(
                    "INSERT INTO annotations(copy_id, ann_id, pos, start, end, "
                    "label, holder) VALUES(?, ?, ?, ?, ?, ?, NULL)",
                    [(copy_id, a["id"], i, a["start"], a["end"], a["label"])
                     for i, a in enumerate(annotations)])
                state = self._load(conn, copy_id)
                self._insert_revision(conn, state, "create", None)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()
        return self.get(copy_id)

    def get(self, copy_id: str) -> dict:
        """读取同一提交状态的快照。"""
        with self._lock:
            conn = self._connect()
            try:
                return self._load(conn, copy_id).snapshot()
            finally:
                conn.close()

    def list_revisions(self, copy_id: str) -> list[dict]:
        """按 revision 升序列出版本及操作信息（不改变当前状态）。

        每项为该版本的完整快照（文本、批注原排序、持有者、status、连通组）
        附 ``operation`` 与 ``reviewer``；未知副本抛 `CopyNotFound`（404）。
        """
        with self._lock:
            conn = self._connect()
            try:
                if conn.execute("SELECT 1 FROM copies WHERE id=?",
                                (copy_id,)).fetchone() is None:
                    raise CopyNotFound(copy_id)
                rows = conn.execute(
                    "SELECT payload FROM copy_revisions "
                    "WHERE copy_id=? ORDER BY revision ASC",
                    (copy_id,)).fetchall()
                return [self._revision_view(json.loads(r[0])) for r in rows]
            finally:
                conn.close()

    def get_revision(self, copy_id: str, revision: int) -> dict:
        """读取指定 revision 的完整快照（不改变当前状态）。

        未知副本 → `CopyNotFound`（404）；副本存在但该版本未记录 →
        `RevisionNotFound`（404，如旧库只补了当前 revision 的 baseline，
        更早的版本不编造、读取返回 404）。
        """
        with self._lock:
            conn = self._connect()
            try:
                if conn.execute("SELECT 1 FROM copies WHERE id=?",
                                (copy_id,)).fetchone() is None:
                    raise CopyNotFound(copy_id)
                row = conn.execute(
                    "SELECT payload FROM copy_revisions "
                    "WHERE copy_id=? AND revision=?",
                    (copy_id, revision)).fetchone()
                if row is None:
                    raise RevisionNotFound(copy_id, revision)
                return self._revision_view(json.loads(row[0]))
            finally:
                conn.close()

    def _apply(self, copy_id: str, operation, op_name: str,
               reviewer: str | None) -> dict:
        """在锁内完整执行「读入 → 运算 → 整体写回」，失败不写任何内容。

        发生 revision 递增的成功提交时，副本变更与该版本完整快照在同一
        事务内写入（``op_name`` / ``reviewer`` 记录操作类型与审阅者）；
        幂等领取（``changed is False``）与一切抛错都不产生版本记录。
        """
        with self._lock:
            conn = self._connect()
            try:
                state = self._load(conn, copy_id)
                changed = operation(state)  # 抛错则直接跳出，不写库
                if changed is not False:     # 幂等领取（False）不产生新版本
                    conn.execute(
                        "UPDATE copies SET revision=?, status=? WHERE id=?",
                        (state.revision, state.status, copy_id))
                    # 整体写回该副本全部批注：持锁读入的最新状态上做的修改
                    # 不会覆盖其他批注（其他请求的修改已在本操作之前提交）。
                    conn.executemany(
                        "UPDATE annotations SET start=?, end=?, label=?, holder=? "
                        "WHERE copy_id=? AND ann_id=?",
                        [(state.annotations[i]["start"],
                          state.annotations[i]["end"],
                          state.annotations[i]["label"],
                          state.holders.get(i), copy_id, i)
                         for i in state.order])
                    self._insert_revision(conn, state, op_name, reviewer)
                    conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()
        return self.get(copy_id)

    def claim(self, copy_id: str, ann_id: str, reviewer: str) -> dict:
        return self._apply(copy_id, lambda st: st.claim(ann_id, reviewer),
                           "claim", reviewer)

    def release(self, copy_id: str, ann_id: str, reviewer: str) -> dict:
        return self._apply(copy_id, lambda st: st.release(ann_id, reviewer),
                           "release", reviewer)

    def modify(self, copy_id: str, ann_id: str, reviewer: str,
               start: int, end: int, label: str) -> dict:
        return self._apply(
            copy_id,
            lambda st: st.modify(ann_id, reviewer, start, end, label),
            "modify", reviewer)

    def finalize(self, copy_id: str, expected_revision: int) -> dict:
        """结稿：与领取 / 释放 / 修改在同一进程锁内串行，等价于某个先后次序。"""
        return self._apply(
            copy_id, lambda st: st.finalize(expected_revision),
            "finalize", None)
