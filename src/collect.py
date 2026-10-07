"""Collect the two frozen ISIC diagnosis_2 classes and split by pHash group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
MANIFEST = DATA / "manifest.csv"
SNAPSHOT = DATA / "api_meta_snapshot.jsonl"
STATE = DATA / "collection_state.json"
API = "https://api.isic-archive.com/api/v2/images/"
DIAGNOSES = {
    "Benign melanocytic proliferations": 0,
    "Malignant melanocytic proliferations (Melanoma)": 1,
}
COLUMNS = [
    "isic_id", "diagnosis_1", "diagnosis_2", "diagnosis_3", "label",
    "image_url", "path", "bytes", "sha256", "phash", "group_id",
    "split", "download_status",
]
CAP_BYTES = 900 * 1024 * 1024


def data_bytes() -> int:
    return sum(path.stat().st_size for path in DATA.rglob("*") if path.is_file())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_manifest(rows: list[dict]) -> None:
    temporary = MANIFEST.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(MANIFEST)


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Existing manifest columns do not match frozen schema")
        return list(reader)


def request_with_retry(session: requests.Session, url: str, *, params=None,
                       stream=False, timeout=(15, 45), stats: Counter):
    for attempt in range(4):
        try:
            response = session.get(url, params=params, stream=stream, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            stats["retries"] += int(attempt < 3)
            if attempt == 3:
                stats["request_failures"] += 1
                raise RuntimeError(f"Request failed after 3 retries: {url}: {exc}") from exc
            time.sleep(0.8 * 2**attempt)


def phash(image_bytes: bytes) -> str:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image cannot be decoded")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(small))[:8, :8]
    threshold = np.median(coefficients.flatten()[1:])
    bits = (coefficients >= threshold).flatten()
    value = sum(int(bit) << (63 - index) for index, bit in enumerate(bits))
    return f"{value:016x}"


def download_image(session: requests.Session, record: dict, rows: list[dict],
                   hashes: set[str], stats: Counter, cap_bytes: int) -> str:
    clinical = record["metadata"]["clinical"]
    isic_id = record["isic_id"]
    url = record.get("files", {}).get("full", {}).get("url")
    entry = {
        "isic_id": isic_id,
        "diagnosis_1": clinical.get("diagnosis_1") or "",
        "diagnosis_2": clinical["diagnosis_2"],
        "diagnosis_3": clinical.get("diagnosis_3") or "",
        "label": DIAGNOSES[clinical["diagnosis_2"]],
        "image_url": url or "",
        "path": "", "bytes": "", "sha256": "", "phash": "",
        "group_id": "", "split": "", "download_status": "failed",
    }
    if not url:
        rows.append(entry)
        stats["download_failures"] += 1
        return "failed"
    advertised = record.get("files", {}).get("full", {}).get("size") or 0
    if data_bytes() + int(advertised) > cap_bytes:
        return "capacity"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    destination = IMAGES / f"{isic_id}{suffix}"
    try:
        base_size = data_bytes()
        for attempt in range(4):
            try:
                with session.get(url, stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=65536):
                        content.extend(chunk)
                        if base_size + len(content) > cap_bytes:
                            return "capacity"
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                stats["retries"] += 1
                time.sleep(0.8 * 2**attempt)
        payload = bytes(content)
        image_hash = hashlib.sha256(payload).hexdigest()
        if image_hash in hashes:
            stats["exact_duplicates"] += 1
            return "duplicate"
        image_phash = phash(payload)
        with destination.open("wb") as handle:
            handle.write(payload)
        entry.update({
            "path": destination.relative_to(ROOT).as_posix(),
            "bytes": len(payload), "sha256": image_hash,
            "phash": image_phash, "download_status": "ok",
        })
        rows.append(entry)
        hashes.add(image_hash)
        return "ok"
    except (requests.RequestException, RuntimeError, OSError, ValueError) as exc:
        stats["download_failures"] += 1
        stats["last_error"] = str(exc)[:250]
        rows.append(entry)
        return "failed"


def assign_groups(rows: list[dict], seed: int) -> tuple[int, int]:
    """Connect every pHash pair within Hamming distance six, then split groups."""
    valid = [row for row in rows if row["download_status"] == "ok"]
    parent = list(range(len(valid)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    values = [int(row["phash"], 16) for row in valid]
    pairs = 0
    for left, value in enumerate(values):
        for right in range(left + 1, len(values)):
            if (value ^ values[right]).bit_count() <= 6:
                union(left, right)
                pairs += 1

    groups = defaultdict(list)
    for index, row in enumerate(valid):
        groups[find(index)].append(row)
    ordered = sorted(groups.values(), key=lambda group: min(row["isic_id"] for row in group))
    for group_index, group in enumerate(ordered, 1):
        group_id = f"g{group_index:04d}"
        for row in group:
            row["group_id"] = group_id

    rng = random.Random(seed)
    rng.shuffle(ordered)
    labels = (0, 1)
    names = ("train", "val", "test")
    fractions = {"train": 0.6, "val": 0.2, "test": 0.2}
    totals = Counter(int(row["label"]) for row in valid)
    targets = {(name, label): totals[label] * fractions[name]
               for name in names for label in labels}
    assigned = Counter()
    for group in ordered:
        group_counts = Counter(int(row["label"]) for row in group)

        def cost(name: str) -> float:
            return sum(
                ((assigned[name, label] + group_counts[label] - targets[name, label]) ** 2
                 - (assigned[name, label] - targets[name, label]) ** 2)
                / max(targets[name, label], 1)
                for label in labels
            )

        chosen = min(names, key=cost)
        for row in group:
            row["split"] = chosen
        for label in labels:
            assigned[chosen, label] += group_counts[label]
    return len(ordered), pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cursor", default=None)
    parser.add_argument("--target-per-class", type=int, default=400)
    parser.add_argument("--max-minutes", type=float, default=50)
    parser.add_argument("--data-cap-mib", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    DATA.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    cap_bytes = min(args.data_cap_mib * 1024 * 1024, CAP_BYTES)
    rows = read_manifest()
    rows = [row for row in rows if row["download_status"] != "ok"
            or (ROOT / row["path"]).is_file()]
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    cursor = args.cursor if args.cursor is not None else state.get("cursor")
    stats = Counter(state.get("stats", {}))
    hashes = {row["sha256"] for row in rows if row["download_status"] == "ok"}
    seen = {row["isic_id"] for row in rows}
    counts = Counter(int(row["label"]) for row in rows if row["download_status"] == "ok")
    deadline = time.monotonic() + args.max_minutes * 60
    reason = "quota"
    session = requests.Session()
    session.headers.update({"User-Agent": "Q-Derm research collector/1.0"})
    while min(counts[0], counts[1]) < args.target_per_class:
        if time.monotonic() >= deadline:
            reason = "time_limit"
            break
        if data_bytes() >= cap_bytes:
            reason = "capacity_limit"
            break
        params = {"limit": args.limit}
        if cursor:
            params["cursor"] = cursor
        try:
            response = request_with_retry(session, API, params=params, stats=stats)
            with response:
                page = response.json()
        except (RuntimeError, ValueError) as exc:
            stats["last_error"] = str(exc)[:250]
            reason = "api_failure"
            break
        records = page.get("results", [])
        if not records:
            reason = "api_exhausted"
            break
        with SNAPSHOT.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stats["pages"] += 1
        stats["records_seen"] += len(records)
        for record in records:
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            clinical = (record.get("metadata") or {}).get("clinical") or {}
            label = DIAGNOSES.get(clinical.get("diagnosis_2"))
            isic_id = record.get("isic_id")
            if label is None or not isic_id or isic_id in seen or counts[label] >= args.target_per_class:
                continue
            seen.add(isic_id)
            outcome = download_image(session, record, rows, hashes, stats, cap_bytes)
            if outcome == "ok":
                counts[label] += 1
            if outcome == "capacity":
                reason = "capacity_limit"
                break
            if min(counts[0], counts[1]) >= args.target_per_class:
                break
        next_url = page.get("next")
        cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0] if next_url else None
        write_manifest(rows)
        write_json(STATE, {"cursor": cursor, "stats": dict(stats), "stop_reason": reason})
        if stats["pages"] % 10 == 0:
            print(f"pages={stats['pages']} records={stats['records_seen']} benign={counts[0]} melanoma={counts[1]} data_mib={data_bytes()/1048576:.1f}", flush=True)
        if reason != "quota" or min(counts[0], counts[1]) >= args.target_per_class:
            break
        if not next_url:
            reason = "api_exhausted"
            break
    groups, near_pairs = assign_groups(rows, seed=42)
    write_manifest(rows)
    achieved = min(counts[0], counts[1])
    quota = args.target_per_class if achieved >= args.target_per_class else (
        300 if achieved >= 300 else 250 if achieved >= 250 else achieved
    )
    if achieved < args.target_per_class:
        reason = f"{reason}; downgraded_target={quota}+{quota}"
    write_json(STATE, {"cursor": cursor, "stats": dict(stats), "stop_reason": reason,
                       "quota": quota, "counts": dict(counts), "groups": groups,
                       "near_duplicate_pairs": near_pairs})
    splits = Counter((row["split"], int(row["label"])) for row in rows if row["download_status"] == "ok")
    print(f"FINAL benign={counts[0]} melanoma={counts[1]} quota={quota}+{quota} groups={groups} near_pairs={near_pairs}")
    print(f"SPLITS {dict(splits)}")
    print(f"DATA_MIB={data_bytes()/1048576:.3f} retries={stats['retries']} download_failures={stats['download_failures']} exact_duplicates={stats['exact_duplicates']} reason={reason}")


if __name__ == "__main__":
    main()
