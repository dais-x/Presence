#!/usr/bin/env python3
"""
Generate pure Tâi-lô romanized Taigi transcripts simulating dementia speech at CDR 0-3.
Distribution: CDR0=350, CDR0.5=300, CDR1=200, CDR2=100, CDR3=50 (total=1000)

Usage:
  python generate_taigi_romanized.py --model qwen2.5:14b
  python generate_taigi_romanized.py --cdr-counts "0:350,0.5:300,1:200,2:100,3:50"
"""
from __future__ import annotations
import argparse, csv, json, random, re, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any
import requests

from taigi_romanized_config import (
    CDR_LEVELS, DEFAULT_CDR_COUNTS, SCENARIOS, SCENARIO_WEIGHTS,
    CDR_STYLE, CDR_PROFILE, CDR_RULES, SYSTEM_PROMPT,
    SPEAKER_GROUPS, ALLOWED_MARKERS, BAD_PATTERNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "transcripts" / "taigi_romanized"


def cdr_dir(cdr: float) -> str:
    return f"cdr_{str(cdr).replace('.', '_')}"


def safe_randint(lo: int, hi: int) -> int:
    lo, hi = int(lo), int(hi)
    if hi < lo: lo, hi = hi, lo
    return random.randint(lo, hi)


def build_prompt(cdr: float, scenario_key: str, speaker_desc: str) -> str:
    style = CDR_STYLE[cdr]
    scenario = SCENARIOS[scenario_key]
    prof = CDR_PROFILE[cdr]

    return f"""Tshiánn sán-sing tsi̍t-tuānn Tâi-uân tiúnn-puè kháu-gí tsia̍t-jī-kó, 100% iōng Tâi-lô lô-má-jī.

CDR level: {cdr} ({style['label']})
Speaker: {speaker_desc}
Scenario: {scenario}
Cognitive-linguistic features: {style['features']}
Length: {style['length']}
Expected pauses: {style['pause_markers']}

{CDR_RULES[cdr]}

Common requirements:
- Output ONLY Tâi-lô romanized text. Absolutely NO Chinese characters (漢字).
- Use natural fillers: enn, ah, hit-ê, tō-sī, honn, lah, leh.
- Brackets only for: [thîng], [tn̂g-thîng], [tiām-tsīng], [thàn-khùi], [ka-sàu].
- Sound like a real elderly person chatting, not a textbook.
- Topic jumps should be natural elderly association, not random.
- Do NOT add titles, numbering, quotes, explanations, or metadata.

Output ONLY the transcript."""


def clean_text(text: str) -> str:
    """Clean model output, preserve allowed markers."""
    # Remove any lines that look like metadata/labels
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are labels/headers
        if stripped.startswith(('#', '**', '##', 'CDR', 'Speaker', 'Scenario')):
            continue
        if ':' in stripped[:20] and len(stripped) < 60 and not any(m in stripped for m in ALLOWED_MARKERS):
            continue
        lines.append(stripped.strip('"\'「」'))
    text = ' '.join(lines)

    # Protect allowed markers
    placeholders = {}
    for i, m in enumerate(ALLOWED_MARKERS):
        ph = f"§§{i}§§"
        placeholders[ph] = m
        text = text.replace(m, ph)

    # Remove any other brackets
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Restore markers
    for ph, m in placeholders.items():
        text = text.replace(ph, m)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_chinese(text: str) -> bool:
    """Check if text contains CJK characters (Chinese/Japanese/Korean)."""
    clean = text
    for m in ALLOWED_MARKERS:
        clean = clean.replace(m, '')
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', clean))


def marker_to_event(marker: str, cdr: float) -> dict:
    prof = CDR_PROFILE[cdr]
    lo, hi = prof["pause_range_ms"]
    if marker == "[thîng]":
        return {"type": "pause", "duration_ms": safe_randint(max(200, lo), min(1200, hi))}
    if marker == "[tn̂g-thîng]":
        return {"type": "pause", "duration_ms": safe_randint(max(1200, lo) if hi >= 1200 else lo, hi)}
    if marker == "[tiām-tsīng]":
        return {"type": "silence", "duration_ms": safe_randint(max(1500, lo) if hi >= 1500 else lo, hi)}
    if marker == "[thàn-khùi]":
        return {"type": "sigh", "duration_ms": safe_randint(600, 1400)}
    if marker == "[ka-sàu]":
        return {"type": "cough", "duration_ms": safe_randint(400, 1200)}
    return {"type": "pause", "duration_ms": safe_randint(lo, hi)}


def text_to_events(text: str, cdr: float) -> list[dict]:
    pattern = r'(\[thîng\]|\[tn̂g-thîng\]|\[tiām-tsīng\]|\[thàn-khùi\]|\[ka-sàu\])'
    parts = re.split(pattern, text)
    events = []
    for p in parts:
        if not p.strip():
            continue
        if p in ALLOWED_MARKERS:
            events.append(marker_to_event(p, cdr))
        else:
            cleaned = p.strip()
            if cleaned:
                events.append({"type": "speech", "text": cleaned})
    # Merge adjacent speech
    merged = []
    for e in events:
        if e["type"] == "speech" and merged and merged[-1]["type"] == "speech":
            merged[-1]["text"] += " " + e["text"]
        else:
            merged.append(e)
    return merged


def events_to_text(events: list[dict]) -> str:
    pieces = []
    for e in events:
        if e["type"] == "speech":
            pieces.append(e["text"])
        elif e["type"] == "pause":
            pieces.append("[tn̂g-thîng]" if e.get("duration_ms", 0) >= 1500 else "[thîng]")
        elif e["type"] == "silence":
            pieces.append("[tiām-tsīng]")
        elif e["type"] == "sigh":
            pieces.append("[thàn-khùi]")
        elif e["type"] == "cough":
            pieces.append("[ka-sàu]")
    return " ".join(pieces)


def event_stats(events: list[dict]) -> dict:
    pauses = [e for e in events if e["type"] in {"pause", "silence"}]
    speeches = [e for e in events if e["type"] == "speech"]
    pause_ms = sum(e.get("duration_ms", 0) for e in pauses)
    speech_chars = sum(len(e.get("text", "")) for e in speeches)
    return {
        "event_count": len(events),
        "speech_count": len(speeches),
        "pause_count": len(pauses),
        "sigh_count": sum(1 for e in events if e["type"] == "sigh"),
        "cough_count": sum(1 for e in events if e["type"] == "cough"),
        "pause_total_ms": pause_ms,
        "mean_pause_ms": round(pause_ms / len(pauses), 2) if pauses else 0,
        "speech_chars": speech_chars,
    }


def infer_labels(text: str) -> dict:
    return {
        "word_finding": sum(text.count(t) for t in ["siūnn-bē-khí", "bē-kì-tit", "m̄-tsai", "hit-ê", "kiò siánn"]),
        "repetition": len(re.findall(r'(\b\S{3,}\b)\s+\1', text)),
        "filler_count": sum(text.count(t) for t in ["enn", "ah", "honn", "lah", "hioh"]),
    }


def validate(cdr: float, text: str, events: list[dict]) -> tuple[bool, str]:
    if not text or len(text) < 15:
        return False, "text too short"
    if not events or not any(e["type"] == "speech" for e in events):
        return False, "no speech events"
    if has_chinese(text):
        return False, "contains Chinese characters"
    stats = event_stats(events)
    pc = stats["pause_count"]
    mn, mx = CDR_PROFILE[cdr]["pause_count"]
    if cdr == 0 and pc > 2:
        return False, f"too many pauses for CDR 0: {pc}"
    if pc < max(0, mn - 1):
        return False, f"too few pauses for CDR {cdr}: {pc}"
    if cdr == 3 and stats["speech_count"] > 12:
        return False, f"too many speech chunks for CDR 3"
    if cdr == 0 and any(m in text for m in ["[tn̂g-thîng]", "[tiām-tsīng]"]):
        return False, "severe markers in CDR 0"
    return True, "ok"


def ollama_chat(url: str, model: str, system: str, user: str, temp: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temp, "top_p": 0.9, "num_predict": 1300},
    }
    r = requests.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()


def verify_ollama(url: str, model: str):
    r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=10)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    if model not in models:
        raise RuntimeError(f"Model {model!r} not found. Available: {', '.join(models)}")


def parse_cdr_counts(val: str) -> dict[float, int]:
    if not val:
        return dict(DEFAULT_CDR_COUNTS)
    counts = {}
    for chunk in val.split(','):
        k, v = chunk.strip().split(':')
        cdr = float(k.strip())
        assert cdr in CDR_LEVELS, f"Invalid CDR: {cdr}"
        counts[cdr] = int(v.strip())
    for c in CDR_LEVELS:
        counts.setdefault(c, 0)
    return counts


def weighted_cycle(weights: dict[str, int], total: int, rng: random.Random) -> list[str]:
    if total <= 0:
        return []
    wsum = sum(weights.values())
    raw = {k: (v / wsum) * total for k, v in weights.items()}
    counts = {k: int(raw[k]) for k in weights}
    rem = total - sum(counts.values())
    frac = sorted(weights, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in frac[:rem]:
        counts[k] += 1
    items = []
    for k, c in counts.items():
        items.extend([k] * c)
    rng.shuffle(items)
    return items


def build_speakers(count: int, seed: int) -> list[dict]:
    base = []
    for gname, gc, desc in SPEAKER_GROUPS:
        for _ in range(gc):
            base.append({"group": gname, "description": desc})
    speakers = []
    for i in range(count):
        t = base[i % len(base)]
        speakers.append({"speaker_id": f"spk_{i+1:03d}", "group": t["group"], "description": t["description"]})
    rng = random.Random(seed)
    order = speakers[:]
    rng.shuffle(order)
    te = round(count * 0.70)
    ve = te + round(count * 0.15)
    splits = {}
    for i, s in enumerate(order):
        splits[s["speaker_id"]] = "train" if i < te else ("val" if i < ve else "test")
    for s in speakers:
        s["split"] = splits[s["speaker_id"]]
    return speakers


def build_plan(args) -> list[dict]:
    rng = random.Random(args.seed)
    cdr_counts = parse_cdr_counts(args.cdr_counts)
    speakers = build_speakers(args.speaker_count, args.seed)
    total = sum(cdr_counts.values())
    scenarios = weighted_cycle(SCENARIO_WEIGHTS, total, rng)
    plan = []
    gi = 0
    for cdr in CDR_LEVELS:
        for idx in range(cdr_counts.get(cdr, 0)):
            sp = speakers[gi % len(speakers)]
            sc = scenarios[gi]
            sid = f"{cdr_dir(cdr)}_{sp['speaker_id']}_{sc}_{idx:04d}"
            plan.append({"sample_id": sid, "cdr": cdr, "scenario": sc, "speaker": sp, "split": sp["split"]})
            gi += 1
    return plan


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["sample_id"]] = rec
    return records


def generate_one(args, cdr: float, scenario: str, speaker: dict) -> tuple[str, list, dict, str, str]:
    prompt = build_prompt(cdr, scenario, speaker["description"])
    last_reason = "not generated"
    text, events, raw = "", [], ""
    for attempt in range(1, args.max_retries + 2):
        raw = ollama_chat(args.url, args.model, SYSTEM_PROMPT, prompt, args.temperature)
        text = clean_text(raw)
        events = text_to_events(text, cdr)
        if events:
            text = events_to_text(events)
        ok, reason = validate(cdr, text, events)
        if ok:
            return text, events, infer_labels(text), raw, "ok"
        last_reason = reason
        prompt = build_prompt(cdr, scenario, speaker["description"]) + f"\n\nPrevious attempt rejected: {reason}. Please regenerate following CDR {cdr} requirements strictly."
    return text, events, infer_labels(text), raw, f"failed: {last_reason}"


def write_outputs(out_dir: Path, records: list[dict], seed: int):
    records = sorted(records, key=lambda r: (r.get("split", ""), float(r["cdr_level"]), r["sample_id"]))

    # metadata.jsonl
    (out_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    # manifest.csv
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "sample_id", "cdr_level", "cdr_label", "scenario", "speaker_id",
            "speaker_group", "split", "text_path", "events_path",
            "generation_time_sec", "validation_status",
        ], extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # splits
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "val": [], "test": []}
    for r in records:
        by_split.setdefault(r.get("split", "train"), []).append(r)
    for sn in ["train", "val", "test"]:
        items = sorted(by_split.get(sn, []), key=lambda r: (float(r["cdr_level"]), r["sample_id"]))
        (split_dir / f"{sn}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")

    # summary
    summary = {"total": len(records), "by_cdr": {}, "by_split": {}, "by_scenario": {}}
    for r in records:
        for f, k in [("by_cdr", str(r["cdr_level"])), ("by_split", r["split"]), ("by_scenario", r["scenario"])]:
            summary[f][k] = summary[f].get(k, 0) + 1
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(args) -> Path:
    verify_ollama(args.url, args.model)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    random.seed(args.seed)
    plan = build_plan(args)
    existing = load_existing(meta_path)
    total = len(plan)

    # Save plan
    (out_dir / "generation_plan.json").write_text(json.dumps({
        "cdr_counts": parse_cdr_counts(args.cdr_counts),
        "speaker_count": args.speaker_count,
        "total_samples": total,
        "model": args.model,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if existing:
        print(f"Resuming {run_id}: {len(existing)}/{total} already exist")

    with meta_path.open("a", encoding="utf-8") as mfp:
        for item in plan:
            sid = item["sample_id"]
            if sid in existing:
                continue

            cdr = item["cdr"]
            ldir = out_dir / cdr_dir(cdr)
            ldir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            text, events, labels, raw, status = generate_one(args, cdr, item["scenario"], item["speaker"])
            elapsed = round(time.time() - t0, 2)

            if not text:
                print(f"EMPTY: {sid}", file=sys.stderr)
                continue

            txt_path = ldir / f"{sid}.txt"
            evt_path = ldir / f"{sid}.events.json"

            txt_path.write_text(text + "\n", encoding="utf-8")

            evt_payload = {
                "sample_id": sid, "cdr_level": cdr, "cdr_label": CDR_STYLE[cdr]["label"],
                "scenario": item["scenario"], "speaker": item["speaker"], "split": item["split"],
                "transcript": text, "event_script": events,
                "labels": labels, "event_stats": event_stats(events),
                "speech_rate_target": CDR_PROFILE[cdr]["speech_rate"],
            }
            evt_path.write_text(json.dumps(evt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            record = {
                "sample_id": sid, "cdr_level": cdr, "cdr_label": CDR_STYLE[cdr]["label"],
                "scenario": item["scenario"],
                "speaker_id": item["speaker"]["speaker_id"],
                "speaker_group": item["speaker"]["group"],
                "split": item["split"],
                "model": args.model, "temperature": args.temperature,
                "text_path": str(txt_path.relative_to(PROJECT_ROOT)),
                "events_path": str(evt_path.relative_to(PROJECT_ROOT)),
                "generation_time_sec": elapsed,
                "validation_status": status,
                "raw_text": text, "raw_response": raw,
                "event_script": events, "event_stats": event_stats(events),
                "labels": labels,
                "speech_rate_target": CDR_PROFILE[cdr]["speech_rate"],
            }
            mfp.write(json.dumps(record, ensure_ascii=False) + "\n")
            mfp.flush()
            existing[sid] = record

            st = record["event_stats"]
            print(f"{len(existing)}/{total} | {sid} | split={record['split']} | "
                  f"{elapsed}s | {len(text)}ch | pauses={st['pause_count']} | {status}")

    write_outputs(out_dir, list(existing.values()), args.seed)
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Generate pure Tâi-lô romanized Taigi CDR transcripts.")
    p.add_argument("--cdr-counts", default="0:350,0.5:300,1:200,2:100,3:50")
    p.add_argument("--speaker-count", type=int, default=40)
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.68)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default="")
    p.add_argument("--max-retries", type=int, default=2)
    args = p.parse_args()
    out = generate(args)
    print(f"\nDataset saved to: {out}")


if __name__ == "__main__":
    main()
