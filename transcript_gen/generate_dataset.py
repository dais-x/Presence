"""
Large-scale CDR-labeled dataset generator.

Generates transcripts via Qwen/Ollama, extracts linguistic features,
exports to CSV, and splits into train/val/test sets.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

# Fix imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_full_prompt, SCENARIOS, DIVERSITY_MODIFIERS
from transcript_parser import parse_transcript

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
CDR_LEVELS = [0, 0.5, 1, 2, 3]


# ── Feature extraction ──────────────────────────────────────────────

def extract_features(raw_text: str, parsed) -> dict:
    """Extract linguistic features from a transcript for the CSV dataset."""
    # Basic text stats
    clean_text = re.sub(r'\[.*?\]', '', raw_text)
    clean_text = re.sub(r'\s+', '', clean_text)

    # Use segments from parsed object for language stats
    zh_segments = [s for s in parsed.segments if s.type == "ZH"]
    nan_segments = [s for s in parsed.segments if s.type == "NAN"]
    
    zh_chars = sum(len(s.content) for s in zh_segments)
    nan_chars = sum(len(s.content) for s in nan_segments)
    total_chars = zh_chars + nan_chars

    # Pause counts
    short_pauses = len(re.findall(r'\[停頓\]', raw_text))
    long_pauses = len(re.findall(r'\[長停頓\]', raw_text))
    silences = len(re.findall(r'\[沉默\]', raw_text))
    sighs = len(re.findall(r'\[嘆氣\]', raw_text))
    coughs = len(re.findall(r'\[咳嗽\]', raw_text))

    # Hesitation markers
    hesitations = len(re.findall(r'那個', raw_text))
    fillers_um = len(re.findall(r'嗯', raw_text))
    fillers_uh = len(re.findall(r'呃', raw_text))

    # Repetitions (same 2+ char phrase repeated within 20 chars)
    repetitions = len(re.findall(r'(.{2,6})(?:.{0,20})\1', raw_text))

    # Ellipsis / trailing off
    trailing = len(re.findall(r'\.{2,}|…', raw_text))

    total_pauses = short_pauses + long_pauses + silences

    return {
        "total_chars": total_chars,
        "mandarin_chars": zh_chars,
        "taiwanese_chars": nan_chars,
        "mandarin_ratio": round(zh_chars / total_chars, 4) if total_chars > 0 else 0,
        "taiwanese_ratio": round(nan_chars / total_chars, 4) if total_chars > 0 else 0,
        "num_taiwanese_segments": len(nan_segments),
        "num_language_switches": parsed.num_switches,
        "short_pauses": short_pauses,
        "long_pauses": long_pauses,
        "silences": silences,
        "total_pauses": total_pauses,
        "pause_density": round(total_pauses / max(total_chars, 1) * 100, 4),
        "sighs": sighs,
        "coughs": coughs,
        "hesitation_nage": hesitations,
        "filler_um": fillers_um,
        "filler_uh": fillers_uh,
        "total_fillers": hesitations + fillers_um + fillers_uh,
        "filler_density": round((hesitations + fillers_um + fillers_uh) / max(total_chars, 1) * 100, 4),
        "repetitions": repetitions,
        "trailing_off": trailing,
    }


# ── Ollama generator (simplified) ───────────────────────────────────

def ollama_generate(system_prompt: str, user_prompt: str,
                    model: str = "qwen2.5:14b",
                    base_url: str = "http://localhost:11434") -> str:
    """Generate a single transcript via Ollama."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.85, "top_p": 0.92, "num_predict": 1024},
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""


# ── Parallel Dataset generation ──────────────────────────────────────

def generate_sample(i, cdr, scenario, model, base_url, timestamp, raw_fp, fp_lock, stats_lock, progress_data):
    """Worker function for a single sample generation."""
    modifiers = list(DIVERSITY_MODIFIERS.keys())
    
    num_mods = random.randint(0, 2)
    selected_mods = random.sample(modifiers, min(num_mods, len(modifiers)))

    # Determine speaker demographics
    gender = random.choice(["male_speaker", "female_speaker"])
    age = random.randint(65, 85)

    # Build prompt with gender modifier
    all_mods = selected_mods + [gender]
    prompt_data = build_full_prompt(cdr, scenario, all_mods)

    # Generate
    start = time.time()
    raw_text = ollama_generate(prompt_data["system"], prompt_data["user"], model, base_url)
    elapsed = time.time() - start

    if not raw_text:
        return None

    sample_id = f"cdr{str(cdr).replace('.','_')}_{scenario}_{timestamp}_{i:04d}"

    # Parse
    parsed = parse_transcript(raw_text, sample_id, cdr, scenario)

    # Extract features
    feats = extract_features(raw_text, parsed)

    # Build row
    row = {
        "sample_id": sample_id,
        "cdr_level": cdr,
        "cdr_label": {0: "normal", 0.5: "very_mild", 1: "mild", 2: "moderate", 3: "severe"}[cdr],
        "binary_label": 0 if cdr == 0 else 1,
        "scenario": scenario,
        "gender": gender.replace("_speaker", ""),
        "age": age,
        "modifiers": "|".join(selected_mods),
        "raw_text": raw_text,
        "generation_time_sec": round(elapsed, 1),
    }
    row.update(feats)

    # Thread-safe write to raw backup
    with fp_lock:
        raw_fp.write(json.dumps({
            "transcript_id": sample_id, "cdr_level": cdr,
            "scenario": scenario, "modifiers": all_mods,
            "raw_text": raw_text, "generation_time_sec": round(elapsed, 1),
        }, ensure_ascii=False) + "\n")
        raw_fp.flush()

    with stats_lock:
        progress_data['count'] += 1
        curr_count = progress_data['count']
        total = progress_data['total']
        logger.info(
            f"  [{curr_count}/{total}] CDR {cdr} | {scenario} | "
            f"ZH:{feats['mandarin_ratio']:.0%} NAN:{feats['taiwanese_ratio']:.0%} | "
            f"switches:{feats['num_language_switches']} pauses:{feats['total_pauses']} | "
            f"{elapsed:.1f}s"
        )

    return row

def generate_dataset_parallel(samples_per_level: int, model: str, base_url: str, concurrency: int) -> list[dict]:
    """Generate the full labeled dataset using multiple threads."""
    scenarios = list(SCENARIOS.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_dir = OUTPUT_DIR / "transcripts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"raw_dataset_{timestamp}.jsonl"
    
    total = samples_per_level * len(CDR_LEVELS)
    all_rows = []
    
    progress_data = {'count': 0, 'total': total}
    fp_lock = threading.Lock()
    stats_lock = threading.Lock()

    logger.info(f"Starting parallel generation with concurrency={concurrency}")

    with open(raw_file, "w", encoding="utf-8") as raw_fp:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for cdr in CDR_LEVELS:
                for i in range(samples_per_level):
                    scenario = scenarios[i % len(scenarios)]
                    futures.append(executor.submit(
                        generate_sample, i, cdr, scenario, model, base_url, 
                        timestamp, raw_fp, fp_lock, stats_lock, progress_data
                    ))
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_rows.append(res)

    logger.info(f"\nRaw backup saved to {raw_file}")
    logger.info(f"Total samples generated: {len(all_rows)}/{total}")
    return all_rows


# ── CSV export & split ───────────────────────────────────────────────

CSV_COLUMNS = [
    "sample_id", "cdr_level", "cdr_label", "binary_label",
    "scenario", "gender", "age", "modifiers",
    "total_chars", "mandarin_chars", "taiwanese_chars",
    "mandarin_ratio", "taiwanese_ratio",
    "num_taiwanese_segments", "num_language_switches",
    "short_pauses", "long_pauses", "silences", "total_pauses", "pause_density",
    "sighs", "coughs",
    "hesitation_nage", "filler_um", "filler_uh", "total_fillers", "filler_density",
    "repetitions", "trailing_off",
    "raw_text", "generation_time_sec",
]


def save_csv(rows: list[dict], filepath: Path):
    """Save rows as CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows → {filepath}")


def split_dataset(rows: list[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified split by CDR level into train/val/test."""
    random.seed(seed)

    # Group by CDR level
    by_cdr = {}
    for row in rows:
        cdr = row["cdr_level"]
        by_cdr.setdefault(cdr, []).append(row)

    train, val, test = [], [], []

    for cdr, samples in sorted(by_cdr.items()):
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(samples[:n_train])
        val.extend(samples[n_train:n_train + n_val])
        test.extend(samples[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def print_split_summary(train, val, test):
    """Print a distribution summary of the splits."""
    print(f"\n{'='*70}")
    print(f"{'DATASET SPLIT SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Split':<10} {'Total':>8} | {'CDR 0':>7} {'CDR 0.5':>8} {'CDR 1':>7} {'CDR 2':>7} {'CDR 3':>7}")
    print(f"{'-'*70}")
    for name, data in [("Train", train), ("Val", val), ("Test", test)]:
        counts = {}
        for r in data:
            counts[r["cdr_level"]] = counts.get(r["cdr_level"], 0) + 1
        print(
            f"{name:<10} {len(data):>8} | "
            f"{counts.get(0,0):>7} {counts.get(0.5,0):>8} "
            f"{counts.get(1,0):>7} {counts.get(2,0):>7} {counts.get(3,0):>7}"
        )
    print(f"{'='*70}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate labeled CDR dataset as CSV")
    parser.add_argument("--samples-per-level", type=int, default=200,
                        help="Samples per CDR level (total = 5x this)")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel requests to Ollama")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{args.url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if args.model not in models:
            logger.error(f"Model '{args.model}' not found. Available: {models}")
            sys.exit(1)
        logger.info(f"✅ Ollama connected — model: {args.model}")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        sys.exit(1)

    # Generate
    logger.info(f"Generating {args.samples_per_level} samples × {len(CDR_LEVELS)} CDR levels "
                f"= {args.samples_per_level * len(CDR_LEVELS)} total")
    
    rows = generate_dataset_parallel(args.samples_per_level, args.model, args.url, args.concurrency)

    if not rows:
        logger.error("No samples generated!")
        sys.exit(1)

    # Save full dataset
    csv_dir = OUTPUT_DIR / "csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(rows, csv_dir / f"full_dataset_{timestamp}.csv")
    save_csv(rows, csv_dir / "full_dataset.csv")  # convenience copy

    # Split
    train, val, test = split_dataset(
        rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    save_csv(train, csv_dir / "train.csv")
    save_csv(val, csv_dir / "val.csv")
    save_csv(test, csv_dir / "test.csv")

    print_split_summary(train, val, test)

    # Feature statistics
    print(f"\n{'='*70}")
    print(f"{'FEATURE STATISTICS BY CDR LEVEL':^70}")
    print(f"{'='*70}")
    print(f"{'CDR':<6} {'ZH%':>6} {'NAN%':>6} {'Switches':>9} {'Pauses':>7} "
          f"{'Fillers':>8} {'Chars':>6} {'Reps':>5}")
    print(f"{'-'*70}")
    for cdr in CDR_LEVELS:
        cdr_rows = [r for r in rows if r["cdr_level"] == cdr]
        if not cdr_rows:
            continue
        n = len(cdr_rows)
        print(
            f"{cdr:<6} "
            f"{sum(r['mandarin_ratio'] for r in cdr_rows)/n:>5.0%} "
            f"{sum(r['taiwanese_ratio'] for r in cdr_rows)/n:>5.0%} "
            f"{sum(r['num_language_switches'] for r in cdr_rows)/n:>9.1f} "
            f"{sum(r['total_pauses'] for r in cdr_rows)/n:>7.1f} "
            f"{sum(r['total_fillers'] for r in cdr_rows)/n:>8.1f} "
            f"{sum(r['total_chars'] for r in cdr_rows)/n:>6.0f} "
            f"{sum(r['repetitions'] for r in cdr_rows)/n:>5.1f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
