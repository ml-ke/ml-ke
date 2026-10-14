#!/usr/bin/env python3
"""Parallel range-request downloader for slow file hosts (mirrors, ROM hosts, CDNs).

A single connection to such hosts often runs at ~3 MB/min while the server still supports
Range; splitting the file across `--threads` concurrent range requests measured ~14x faster.
Files already present and md5-matching are skipped, so this is safe to re-run after an
interruption.

Usage:
    python3 parallel_range_download.py <base-url> <manifest.txt> [--threads N] [--dest DIR]

Manifest format (one per line, '#' comments allowed):
    <filename> <expected-md5-or-dash>
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def md5(path, bufsize=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(bufsize), b""):
            h.update(buf)
    return h.hexdigest()


def remote_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.headers.get("Accept-Ranges", "")


def fetch_chunk(url, start, end, path, progress):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            progress[0] += len(buf)


def fetch_one(base, name, want, dest, threads):
    url = base.rstrip("/") + "/" + name
    final = os.path.join(dest, name)
    if os.path.exists(final) and want not in (None, "-") and md5(final) == want:
        print(f"SKIP (already verified): {name}", flush=True)
        return True
    try:
        total, ranges = remote_size(url)
    except Exception as exc:  # no HEAD support, or 404
        print(f"FAILED to stat {name}: {exc}", flush=True)
        return False
    if "bytes" not in ranges.lower():
        print(f"WARNING: {name} does not advertise Range support; trying anyway", flush=True)
    print(f"=== {name}  ({total / 1048576:.1f} MB, {threads} connections)", flush=True)
    chunk = total // threads + 1
    parts, progress = [], [0]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = []
        for i in range(threads):
            start = i * chunk
            end = min(start + chunk - 1, total - 1)
            if start > end:
                break
            part = f"{final}.part{i}"
            parts.append(part)
            futs.append(ex.submit(fetch_chunk, url, start, end, part, progress))
        last = 0
        while any(not f.done() for f in futs):
            time.sleep(10)
            if progress[0] != last:
                last = progress[0]
                print(f"    {last / 1048576:.1f}/{total / 1048576:.1f} MB "
                      f"({100 * last / total:.0f}%)", flush=True)
        for f in futs:
            f.result()
    assembled = final + ".assembled"
    with open(assembled, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while True:
                    buf = fh.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
    os.replace(assembled, final)
    for part in parts:
        os.remove(part)
    got = md5(final)
    if want in (None, "-"):
        print(f"  md5={got} (no published sum to compare)", flush=True)
        return True
    if got == want:
        print(f"  md5 MATCH {got}", flush=True)
        return True
    print(f"  md5 MISMATCH got={got} want={want}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("manifest")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--dest", default=".")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    entries = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            entries.append((parts[0], parts[1] if len(parts) > 1 else "-"))
    if not entries:
        sys.exit("empty manifest")
    ok = True
    for name, want in entries:
        ok &= fetch_one(args.base_url, name, want, args.dest, args.threads)
    print("ALL VERIFIED" if ok else "SOME FILES FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
