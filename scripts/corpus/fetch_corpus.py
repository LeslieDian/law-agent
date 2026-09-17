#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""法律语料采集器（阶段 1）——仅标准库 + pyyaml，兼容 Python 3.8+。

设计原则：
  1. **不硬编码数据集名**：数据集清单从 `configs/corpus_sources.yaml` 读取。
  2. **只按 license_grade 放行**：默认只采 A 级（协议明确，可进论文实验集）。
  3. **体积护栏**：单数据集超过 --max-gb 直接拒绝（防止误拉 MNBVC-judgment 那种 124GB）。
  4. **落盘即登记**：每个文件算 SHA-256，写 `MANIFEST.json` + 每个数据集一份 `_archive/source.json`。
  5. 走 HF_ENDPOINT（默认 https://hf-mirror.com），多线程分块 + 断点续传。

用法：
  python scripts/corpus/fetch_corpus.py                       # 默认：A 级，跳过 case_corpus
  python scripts/corpus/fetch_corpus.py --only ShengbinYue/DISC-Law-SFT
  python scripts/corpus/fetch_corpus.py --grade A --threads 12
  python scripts/corpus/fetch_corpus.py --list                # 只列出将要采什么，不下载
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0 (compatible; law-agent-corpus-fetcher/1.0)"}
CHUNK = 32 * 1024 * 1024
SRC_YAML_DEFAULT = "configs/corpus_sources.yaml"
OUT_ROOT_DEFAULT = "data/corpus/raw"


def log(msg: str) -> None:
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return "{:.1f}{}".format(n, unit)
        n /= 1024
    return "{:.1f}PB".format(n)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def http_open(url: str, headers: dict | None = None, timeout: int = 90, retries: int = 6):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, **(headers or {})}
            )
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # 4xx 除了 429 都不要重试（401 = 不存在或无权，重试无意义）
            if exc.code < 500 and exc.code != 429:
                raise
            last = exc
            log("    请求失败 ({}/{}): {}".format(i + 1, retries, exc))
            time.sleep(min(2 * (i + 1), 15))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log("    请求失败 ({}/{}): {}".format(i + 1, retries, exc))
            time.sleep(min(2 * (i + 1), 15))
    raise last if last else RuntimeError("http_open failed: " + url)


def api_json(url: str, token: str | None = None) -> dict:
    headers = {"Authorization": "Bearer " + token} if token else {}
    with http_open(url, headers) as resp:
        return json.loads(resp.read().decode("utf-8"))


def dataset_info(repo: str, token: str | None = None) -> dict:
    return api_json("{}/api/datasets/{}".format(ENDPOINT, repo), token)


def dataset_tree(repo: str, revision: str = "main", token: str | None = None) -> list:
    """返回 [(path, size, sha256_or_None)]。"""
    url = "{}/api/datasets/{}/tree/{}?recursive=true".format(ENDPOINT, repo, revision)
    data = api_json(url, token)
    out = []
    for item in data:
        if item.get("type") != "file":
            continue
        size = item.get("size") or 0
        oid = None
        lfs = item.get("lfs") or {}
        if lfs.get("size"):
            size = lfs["size"]
        if isinstance(lfs.get("oid"), str) and len(lfs["oid"]) == 64:
            oid = lfs["oid"]
        out.append((item["path"], int(size), oid))
    return out


def resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return "{}/datasets/{}/resolve/{}/{}".format(ENDPOINT, repo, revision, path)


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------

def sha256_of(path: str, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def download_single(url: str, dest: str, size: int, token: str | None = None) -> bool:
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": "bytes={}-".format(have)} if have else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with http_open(url, headers) as resp, open(tmp, "ab" if have else "wb") as fh:
            while True:
                block = resp.read(4 * 1024 * 1024)
                if not block:
                    break
                fh.write(block)
    except Exception as exc:  # noqa: BLE001
        log("    单连接下载失败: {}".format(exc))
        return False
    got = os.path.getsize(tmp)
    if size and got != size:
        log("    大小不符: 期望 {} 实际 {}".format(size, got))
        return False
    os.replace(tmp, dest)
    return True


def download_parallel(url: str, dest: str, size: int, threads: int,
                      token: str | None = None) -> bool:
    with open(dest, "wb") as fh:
        fh.truncate(size)

    chunks = [(o, min(o + CHUNK, size)) for o in range(0, size, CHUNK)]
    lock = threading.Lock()
    queue = list(chunks)
    written = [0]
    failures = []
    no_range = [False]

    def worker() -> None:
        while True:
            with lock:
                if not queue or no_range[0]:
                    return
                start, end = queue.pop(0)
            headers = {"Range": "bytes={}-{}".format(start, end - 1)}
            if token:
                headers["Authorization"] = "Bearer " + token
            for attempt in range(6):
                try:
                    with http_open(url, headers, timeout=180) as resp:
                        status = getattr(resp, "status", 200)
                        if status != 206:
                            with lock:
                                no_range[0] = True
                                queue.clear()
                            return
                        data = resp.read()
                    if len(data) != end - start:
                        raise RuntimeError(
                            "分块长度不符 {} != {}".format(len(data), end - start)
                        )
                    with open(dest, "r+b") as fh:
                        fh.seek(start)
                        fh.write(data)
                    with lock:
                        written[0] += len(data)
                    break
                except Exception as exc:  # noqa: BLE001
                    if no_range[0]:
                        return
                    if attempt == 5:
                        with lock:
                            failures.append("{}..{}: {}".format(start, end, exc))
                    else:
                        time.sleep(1.5 * (attempt + 1))

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for w in workers:
        w.start()
    last_report = time.time()
    while any(w.is_alive() for w in workers):
        time.sleep(2)
        now = time.time()
        if now - last_report >= 20:
            last_report = now
            with lock:
                done = written[0]
            log("    进度 {}/{} ({:.1f}%)".format(
                human(done), human(size), 100.0 * done / max(size, 1)))

    if failures:
        log("    !! {} 个分块失败".format(len(failures)))
        for f in failures[:5]:
            log("       " + f)
        return False
    if os.path.getsize(dest) != size:
        log("    !! 最终大小不符")
        return False
    return True


def fetch_file(repo: str, path: str, size: int, oid, dest_root: str,
               threads: int, token: str | None) -> tuple:
    """返回 (ok, sha256_实际值, 字节数)。"""
    dest = os.path.join(dest_root, path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        actual = sha256_of(dest)
        if oid and actual != oid:
            log("  已有文件 sha256 不符，重下: {}".format(path))
            os.remove(dest)
        else:
            log("  跳过（已在且校验通过）: {}".format(path))
            return True, actual, size

    url = resolve_url(repo, path)
    log("  下载 {} ({})".format(path, human(size) if size else "未知大小"))
    t0 = time.time()
    ok = False
    if size and size > CHUNK:
        ok = download_parallel(url, dest, size, threads, token)
        if not ok:
            log("  多线程失败，回退单连接")
            if os.path.exists(dest):
                os.remove(dest)
            ok = download_single(url, dest, size, token)
    else:
        ok = download_single(url, dest, size, token)

    if not ok:
        log("  !! 失败: {}".format(path))
        return False, None, 0

    actual_size = os.path.getsize(dest)
    speed = actual_size / max(time.time() - t0, 1e-6)
    log("  完成 {} 用时 {:.0f}s，均速 {}/s".format(
        path, time.time() - t0, human(speed)))

    actual = sha256_of(dest)
    if oid:
        if actual == oid:
            log("  sha256 与 LFS oid 校验通过")
        else:
            log("  !! sha256 与 LFS oid 不匹配\n     期望 {}\n     实际 {}".format(oid, actual))
            return False, actual, actual_size
    return True, actual, actual_size


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def slugify(repo: str) -> str:
    return repo.replace("/", "__")


def load_sources(path: str) -> dict:
    import yaml  # pyyaml 已在 envs/main 中（6.0.3）
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select(sources: list, grade: str, only: list, skip: list,
           exclude_usage: list, max_gb: float) -> tuple:
    """按 license_grade / --only / --skip / --exclude-usage / 体积上限 选源。

    注意：**不要按 usage 里的 case_corpus 一刀切**——像 Dusker/lawyer-llama 这种
    usage=[expert_train, case_corpus] 但只有 446MB 的源是合法要采的。
    真正的超大语料（MNBVC-judgment 121GB）由体积上限拦。
    """
    picked, rejected = [], []
    limit = max_gb * 1024 ** 3
    for s in sources:
        sid = s.get("id")
        usage = s.get("usage") or []
        if only and sid not in only:
            continue
        if sid in skip:
            rejected.append((sid, "在 --skip 列表中"))
            continue
        if s.get("license_grade") != grade:
            rejected.append((sid, "license_grade={} 非 {}".format(s.get("license_grade"), grade)))
            continue
        hit = [u for u in usage if u in exclude_usage]
        if hit:
            rejected.append((sid, "--exclude-usage 命中: {}".format(",".join(hit))))
            continue
        declared = int(s.get("size_bytes") or 0)
        if declared > limit:
            rejected.append((sid, "声明体积 {} 超过 --max-gb {} 上限（超大语料只允许按需抽样）".format(
                human(declared), max_gb)))
            continue
        picked.append(s)
    return picked, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=SRC_YAML_DEFAULT, help="候选源登记表")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT, help="落盘根目录")
    ap.add_argument("--manifest", default="data/corpus/MANIFEST.json")
    ap.add_argument("--grade", default="A", help="只采该 license_grade（默认 A）")
    ap.add_argument("--only", nargs="*", default=[], help="只采这些 id")
    ap.add_argument("--skip", nargs="*", default=[], help="跳过这些 id")
    ap.add_argument("--exclude-usage", nargs="*", default=[],
                    help="排除 usage 命中这些标签的源（一般不需要；体积由 --max-gb 兜底）")
    ap.add_argument("--max-gb", type=float, default=3.0,
                    help="单数据集体积上限 GB（护栏，默认 3.0；MNBVC 124GB 会被这里拦下）")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="只列出计划，不下载")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    cfg = load_sources(a.sources)
    picked, rejected = select(cfg.get("sources") or [], a.grade, a.only, a.skip,
                              a.exclude_usage, a.max_gb)

    log("端点            : {}".format(ENDPOINT))
    log("候选源登记表    : {}".format(a.sources))
    log("落盘根目录      : {}".format(a.out_root))
    log("license_grade   : {}".format(a.grade))
    log("HF_TOKEN        : {}".format("已设置" if token else "未设置"))
    log("")
    log("=== 计划采集 {} 个数据集 ===".format(len(picked)))
    plan_total = 0
    for s in picked:
        sz = int(s.get("size_bytes") or 0)
        plan_total += sz
        tag = ""
        if "case_corpus" in (s.get("usage") or []):
            tag = "  [含 case_corpus 用途，仅作训练/知识库，不得进评测]"
        log("  {:<45} {:>10}  {}{}".format(
            s["id"], human(sz), s.get("license") or "?", tag))
    log("  合计约 {}".format(human(plan_total)))
    log("")
    log("=== 已排除 {} 个 ===".format(len(rejected)))
    for sid, why in rejected:
        log("  {:<45} {}".format(sid, why))
    log("")

    if a.list:
        log("MARKER_FETCH_LIST_ONLY")
        return 0

    os.makedirs(a.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)

    manifest = {
        "dataset": "law-agent training corpus (stage-1 raw)",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "endpoint": ENDPOINT,
        "license_grade": a.grade,
        "sources_yaml": a.sources,
        "datasets": [],
        "rejected": [{"id": s, "reason": r} for s, r in rejected],
    }

    grand_ok = True
    for s in picked:
        sid = s["id"]
        declared = int(s.get("size_bytes") or 0)
        log("=" * 72)
        log("数据集: {}  ({} / {})".format(sid, s.get("license") or "?", a.grade))

        if declared > a.max_gb * 1024 ** 3:
            log("  !! 拒绝：声明体积 {} 超过上限 {} GB（护栏生效）".format(
                human(declared), a.max_gb))
            manifest["datasets"].append({
                "id": sid, "status": "SKIPPED_SIZE_GUARD",
                "declared_bytes": declared, "max_gb": a.max_gb,
            })
            continue

        slug = slugify(sid)
        dest_root = os.path.join(a.out_root, slug)

        try:
            info = dataset_info(sid, token)
        except Exception as exc:  # noqa: BLE001
            log("  !! 无法取仓库信息（可能不存在 / 需授权）: {}".format(exc))
            manifest["datasets"].append({
                "id": sid, "status": "INFO_FAILED", "error": str(exc)[:300],
            })
            grand_ok = False
            continue

        commit_sha = info.get("sha")
        gated = info.get("gated")
        log("  commit={}  gated={}".format(commit_sha, gated))

        if gated and gated is not False and not token:
            log("  !! 该仓库为 gated，且未设置 HF_TOKEN → 跳过")
            manifest["datasets"].append({
                "id": sid, "status": "SKIPPED_GATED_NO_TOKEN",
                "commit_sha": commit_sha, "gated": gated,
            })
            grand_ok = False
            continue

        try:
            files = dataset_tree(sid, commit_sha or "main", token)
        except Exception as exc:  # noqa: BLE001
            log("  !! 无法取文件列表: {}".format(exc))
            manifest["datasets"].append({
                "id": sid, "status": "TREE_FAILED",
                "commit_sha": commit_sha, "error": str(exc)[:300],
            })
            grand_ok = False
            continue

        real_total = sum(x[1] for x in files)
        log("  文件 {} 个，实测总体积 {}".format(len(files), human(real_total)))
        if real_total > a.max_gb * 1024 ** 3:
            log("  !! 拒绝：实测体积超过上限 {} GB".format(a.max_gb))
            manifest["datasets"].append({
                "id": sid, "status": "SKIPPED_SIZE_GUARD",
                "commit_sha": commit_sha, "real_bytes": real_total,
                "max_gb": a.max_gb,
            })
            continue

        entry = {
            "id": sid,
            "slug": slug,
            "status": "IN_PROGRESS",
            "hub": s.get("hub", "huggingface"),
            "endpoint": ENDPOINT,
            "url": "{}/datasets/{}".format(ENDPOINT, sid),
            "commit_sha": commit_sha,
            "license": s.get("license"),
            "license_grade": s.get("license_grade"),
            "usage": s.get("usage") or [],
            "declared_bytes": declared,
            "real_bytes": real_total,
            "dest": os.path.abspath(dest_root),
            "fetched_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "files": [],
        }

        failed_files = []
        for path, size, oid in files:
            try:
                ok, actual, nbytes = fetch_file(sid, path, size, oid,
                                                dest_root, a.threads, token)
            except Exception as exc:  # noqa: BLE001
                log("  !! 异常 {}: {}".format(path, exc))
                ok, actual, nbytes = False, None, 0
            if ok:
                entry["files"].append({
                    "path": path, "bytes": nbytes, "sha256": actual,
                    "lfs_oid": oid,
                })
            else:
                failed_files.append(path)

        entry["file_count"] = len(entry["files"])
        entry["total_bytes"] = sum(f["bytes"] for f in entry["files"])
        entry["status"] = "OK" if not failed_files else "PARTIAL_FAILED"
        entry["failed_files"] = failed_files
        manifest["datasets"].append(entry)

        # 每完成一个数据集就落一次盘，便于中断后查看
        # 单独留一份溯源文件
        arc = os.path.join(dest_root, "_archive")
        os.makedirs(arc, exist_ok=True)
        with open(os.path.join(arc, "source.json"), "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in entry.items() if k != "files"},
                      fh, ensure_ascii=False, indent=2)
        with open(a.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

        if failed_files:
            grand_ok = False
            log("  !! 数据集 {} 有 {} 个文件失败".format(sid, len(failed_files)))

    # 汇总
    ok_ds = [d for d in manifest["datasets"] if d.get("status") == "OK"]
    manifest["summary"] = {
        "datasets_ok": len(ok_ds),
        "datasets_total": len(manifest["datasets"]),
        "files": sum(d.get("file_count", 0) for d in ok_ds),
        "total_bytes": sum(d.get("total_bytes", 0) for d in ok_ds),
        "total_human": human(sum(d.get("total_bytes", 0) for d in ok_ds)),
    }
    with open(a.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    log("")
    log("=" * 72)
    log("汇总: {} / {} 个数据集成功，共 {} 文件，{}".format(
        manifest["summary"]["datasets_ok"], manifest["summary"]["datasets_total"],
        manifest["summary"]["files"], manifest["summary"]["total_human"]))
    log("MANIFEST: {}".format(os.path.abspath(a.manifest)))
    for d in manifest["datasets"]:
        log("  {:<45} {}".format(d["id"], d.get("status")))

    log("ALL_OK" if grand_ok else "SOME_FAILED")
    log("MARKER_FETCH_CORPUS_DONE")
    return 0 if grand_ok else 1


if __name__ == "__main__":
    sys.exit(main())
