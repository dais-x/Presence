#!/usr/bin/env python3
"""Create CDR and language-quality audit reports for the Taigi transcripts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata.jsonl"
OUT_DIR = ROOT / "analysis"
OUT_CSV = OUT_DIR / "cdr_transcript_audit.csv"
OUT_MD = OUT_DIR / "cdr_transcript_summary.md"

PAUSE_MARKERS = ("[停頓]", "[長停頓]", "[沉默]", "[嘆氣]", "[咳嗽]")
UNCERTAINTY_TERMS = (
    "毋知",
    "袂知",
    "不知",
    "想袂起來",
    "記袂",
    "袂記得",
    "嗯記得",
    "忘記",
    "不知道",
)
ORIENTATION_TERMS = (
    "幾號",
    "幾月",
    "今仔日",
    "今日",
    "星期",
    "禮拜",
    "月",
    "號",
)
TOPIC_TERMS = ("早餐", "早飯", "食飯", "菜市仔", "家裡", "厝", "阿母", "孫仔")
DRIFT_TERMS = ("講到別處", "攏講", "按呢冷", "冷啦", "麻煩", "好乾淨")
NONSTANDARD_TERMS = ("几", "ㄟ", "𠲎", "嘸", "佷", "海報", "俺", "係")


def folder_level(path: str) -> float:
    name = Path(path).parent.name
    return float(name.removeprefix("cdr_").replace("_", "."))


def clinical_label(level: float) -> str:
    return {
        0.0: "normal",
        0.5: "questionable impairment",
        1.0: "mild dementia",
        2.0: "moderate dementia",
        3.0: "severe dementia",
    }[level]


def md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def sentence_count(text: str) -> int:
    cleaned = re.sub(r"\[[^\]]+\]", "", text)
    return len([part for part in re.split(r"[。！？?]+", cleaned) if part.strip()])


def fragment_ratio(text: str) -> float:
    parts = [part.strip() for part in re.split(r"[。！？?]+", re.sub(r"\[[^\]]+\]", "", text))]
    parts = [part for part in parts if part]
    if not parts:
        return 1.0
    short = sum(1 for part in parts if len(part) <= 8)
    return round(short / len(parts), 3)


def infer_cdr(features: dict[str, float]) -> float:
    long_pause = features["long_pause_count"]
    silence = features["silence_count"]
    uncertainty = features["uncertainty_count"]
    chars = features["char_count"]
    frag = features["fragment_ratio"]

    if chars <= 140 and silence >= 1 and (long_pause >= 1 or chars <= 95):
        return 3.0
    if long_pause + silence >= 5 and frag >= 0.45:
        return 2.0
    if long_pause + silence >= 3 and uncertainty >= 2:
        return 2.0
    if long_pause >= 1 and uncertainty >= 1:
        return 1.0
    if uncertainty >= 1:
        return 0.5
    return 0.0


def score_text(obj: dict) -> dict:
    text = obj["raw_text"]
    level = float(obj["cdr_level"])
    path_level = folder_level(obj["text_path"])
    parsed_meta = obj.get("parsed", {}).get("metadata", {})
    pauses = {marker: text.count(marker) for marker in PAUSE_MARKERS}
    bracket_opens = text.count("[")
    bracket_closes = text.count("]")

    features = {
        "char_count": len(text),
        "sentence_count": sentence_count(text),
        "pause_count": pauses["[停頓]"],
        "long_pause_count": pauses["[長停頓]"],
        "silence_count": pauses["[沉默]"],
        "sigh_count": pauses["[嘆氣]"],
        "cough_count": pauses["[咳嗽]"],
        "uncertainty_count": count_terms(text, UNCERTAINTY_TERMS),
        "orientation_terms": count_terms(text, ORIENTATION_TERMS),
        "topic_terms": count_terms(text, TOPIC_TERMS),
        "drift_terms": count_terms(text, DRIFT_TERMS),
        "fragment_ratio": fragment_ratio(text),
        "taiwanese_ratio": round(float(parsed_meta.get("taiwanese_ratio", 0)), 3),
        "mandarin_ratio": round(float(parsed_meta.get("mandarin_ratio", 0)), 3),
        "num_switches": int(parsed_meta.get("num_switches", 0)),
    }
    inferred = infer_cdr(features)

    flags: list[str] = []
    if level != path_level:
        flags.append("metadata_folder_mismatch")
    if inferred != level:
        flags.append(f"heuristic_cdr_{inferred:g}")
    if bracket_opens != bracket_closes:
        flags.append("unbalanced_pause_marker")
    if features["char_count"] < 20:
        flags.append("too_short")
    if features["taiwanese_ratio"] and features["taiwanese_ratio"] < 0.2:
        flags.append("low_taigi_ratio")
    if features["mandarin_ratio"] > 0.8:
        flags.append("high_mandarin_ratio")
    found_nonstandard = [term for term in NONSTANDARD_TERMS if term in text]
    if found_nonstandard:
        flags.append("orthography_review:" + "|".join(found_nonstandard[:4]))

    if not flags:
        quality = "pass"
    elif any(flag.startswith(("metadata", "unbalanced", "too_short")) for flag in flags):
        quality = "fix"
    else:
        quality = "review"

    return {
        "sample_id": obj["sample_id"],
        "text_path": obj["text_path"],
        "scenario": obj["scenario"],
        "assigned_cdr": f"{level:g}",
        "assigned_label": clinical_label(level),
        "heuristic_cdr": f"{inferred:g}",
        "cdr_alignment": "match" if inferred == level else "review",
        "language_quality": quality,
        "review_flags": ";".join(flags),
        **features,
    }


def main() -> None:
    rows = [score_text(json.loads(line)) for line in METADATA.read_text(encoding="utf-8-sig").splitlines()]
    OUT_DIR.mkdir(exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_level: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_level[row["assigned_cdr"]].append(row)

    lines = [
        "# CDR Transcript Audit Summary",
        "",
        "This audit rates each simulated Taigi transcript against its assigned CDR folder label and flags language/structure issues for review. It is a dataset-quality audit, not a clinical diagnosis.",
        "",
        "## Rubric",
        "",
        "- CDR 0: coherent, oriented response with only normal hesitation.",
        "- CDR 0.5: mild retrieval difficulty or occasional uncertainty, mostly coherent.",
        "- CDR 1: clear memory/orientation problems, shorter answers, repeated uncertainty.",
        "- CDR 2: moderate impairment with fragmented structure, long pauses/silence, topic drift.",
        "- CDR 3: severe impairment with sparse, disconnected phrases and frequent silence.",
        "",
        "## Folder Summary",
        "",
        "| CDR | Label | Files | Heuristic Matches | Review/Fix Flags | Avg chars | Avg Taigi ratio | Avg long pauses+silence |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for level in ("0", "0.5", "1", "2", "3"):
        group = by_level[level]
        matches = sum(1 for row in group if row["cdr_alignment"] == "match")
        flagged = sum(1 for row in group if row["language_quality"] != "pass")
        avg_chars = sum(row["char_count"] for row in group) / len(group)
        avg_taigi = sum(row["taiwanese_ratio"] for row in group) / len(group)
        avg_disfluent = sum(row["long_pause_count"] + row["silence_count"] for row in group) / len(group)
        lines.append(
            f"| {level} | {clinical_label(float(level))} | {len(group)} | {matches} | {flagged} | "
            f"{avg_chars:.1f} | {avg_taigi:.3f} | {avg_disfluent:.2f} |"
        )

    quality_counts = Counter(row["language_quality"] for row in rows)
    alignment_counts = Counter(row["cdr_alignment"] for row in rows)
    flag_counts: Counter[str] = Counter()
    for row in rows:
        for flag in filter(None, row["review_flags"].split(";")):
            if flag.startswith("orthography_review"):
                flag_counts["orthography_review"] += 1
            elif flag.startswith("heuristic_cdr"):
                flag_counts[flag] += 1
            else:
                flag_counts[flag] += 1

    lines.extend(
        [
            "",
            "## Overall Counts",
            "",
            f"- Files audited: {len(rows)}",
            f"- CDR heuristic matches: {alignment_counts['match']}",
            f"- CDR heuristic review cases: {alignment_counts['review']}",
            f"- Language pass: {quality_counts['pass']}",
            f"- Language review: {quality_counts['review']}",
            f"- Language fix: {quality_counts['fix']}",
            "",
            "## Common Review Flags",
            "",
            "| Flag | Count | Meaning |",
            "|---|---:|---|",
        ]
    )
    flag_meanings = {
        "orthography_review": "Contains unusual or inconsistent Taigi orthography/characters that should be checked.",
        "low_taigi_ratio": "Parsed Taigi token ratio is below 0.20.",
        "high_mandarin_ratio": "Parsed Mandarin token ratio is above 0.80.",
    }
    for flag, count in flag_counts.most_common():
        meaning = flag_meanings.get(flag)
        if meaning is None and flag.startswith("heuristic_cdr_"):
            meaning = "Heuristic severity estimate differs from the assigned folder CDR."
        elif meaning is None:
            meaning = "Structural or metadata issue."
        lines.append(f"| `{flag}` | {count} | {meaning} |")

    sample_review = [row for row in rows if row["language_quality"] != "pass"][:20]
    lines.extend(
        [
            "",
            "## First Review Samples",
            "",
            "| sample_id | assigned | heuristic | flags |",
            "|---|---:|---:|---|",
        ]
    )
    for row in sample_review:
        lines.append(
            f"| `{row['sample_id']}` | {row['assigned_cdr']} | {row['heuristic_cdr']} | "
            f"{md_cell(row['review_flags'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The folder label remains the authoritative CDR label for this simulated dataset.",
            "- `heuristic_cdr` is a consistency check based on pauses, silences, uncertainty terms, fragmentation, and transcript length.",
            "- `language_quality=review` usually means Taigi/Mandarin mix, unusual orthography, or a CDR mismatch worth human review.",
            "- `language_quality=fix` is reserved for structural problems such as unbalanced markers, very short text, or metadata/folder mismatch.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
