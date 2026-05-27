#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Taiwanese Mandarin dementia-simulation transcripts for BreezyVoice.

Pipeline:
1. Calls Qwen2.5:14B through Ollama.
2. Generates 1000 Traditional Chinese Taiwanese Mandarin transcripts.
3. Covers CDR distribution:
   CDR 0   = 350
   CDR 0.5 = 300
   CDR 1   = 200
   CDR 2   = 100
   CDR 3   = 50
4. Saves:
   - dataset_metadata.csv      full research metadata + biomarkers
   - breezyvoice_batch.csv     minimal BreezyVoice batch input
   - generation_log.jsonl      raw model outputs / errors

Before running:
    ollama pull qwen2.5:14b
    ollama serve

Run:
    python generate_tw_mandarin_dementia_transcripts.py

Optional:
    python generate_tw_mandarin_dementia_transcripts.py --dry-run
    python generate_tw_mandarin_dementia_transcripts.py --output-dir my_dataset
"""

import argparse
import csv
import json
import random
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional


# -----------------------------
# Dataset distribution
# -----------------------------

CDR_DISTRIBUTION = {
    "0": 350,
    "0.5": 300,
    "1": 200,
    "2": 100,
    "3": 50,
}

LANGUAGE = "Taiwanese Mandarin"
SCRIPT = "Traditional Chinese"

TOPICS = [
    "早上起床後的日常",
    "昨天做了什麼事情",
    "去菜市場買菜",
    "搭公車或捷運出門",
    "和家人一起吃飯",
    "去便利商店或全聯買東西",
    "整理家裡和洗衣服",
    "去診所或醫院拿藥",
    "和朋友聊天",
    "準備午餐或晚餐",
    "回憶上週發生的事情",
    "描述一次出門經驗",
    "說明今天的計畫",
    "談論家裡附近的環境",
    "描述買東西的過程",
]

TAIWAN_MANDARIN_HINTS = [
    "使用自然台灣華語，不要使用中國大陸用語。",
    "可自然使用台灣常見詞彙，例如：捷運、公車、菜市場、便利商店、全聯、健保卡、診所、里長、便當、早餐店。",
    "使用繁體中文。",
    "內容像一般台灣人日常口語敘述，不要像作文、新聞稿或醫學報告。",
]

BANNED_CONTENT = [
    "失智", "阿茲海默", "癡呆", "認知症", "CDR", "臨床", "診斷", "病人", "患者",
    "MMSE", "MoCA", "神經心理", "記憶測驗"
]


# -----------------------------
# CDR simulation rules
# -----------------------------

CDR_RULES = {
    "0": {
        "description": "normal healthy speech",
        "transcript_style": (
            "語句流暢，時間順序清楚，內容具體，少量或沒有停頓詞；"
            "不要出現明顯記憶困難、重複或離題。"
        ),
        "duration_sec_range": (25, 35),
        "biomarker_targets": {
            "hesitation_count": [0, 1],
            "repetition_count": [0, 1],
            "self_correction_count": [0, 1],
            "word_finding_difficulty_count": [0, 1],
            "topic_drift_score": [0.0, 0.1],
            "temporal_disorganization_score": [0.0, 0.1],
            "sentence_fragment_ratio": [0.0, 0.1],
            "information_density_score": [0.75, 1.0],
            "coherence_score": [0.8, 1.0],
        },
    },
    "0.5": {
        "description": "very mild cognitive impairment style speech",
        "transcript_style": (
            "大致仍能完成敘述，但有輕微猶豫、少量重複、偶爾找詞困難，"
            "可能出現「嗯……」「我想一下」「那個」等口語停頓。"
        ),
        "duration_sec_range": (28, 40),
        "biomarker_targets": {
            "hesitation_count": [2, 4],
            "repetition_count": [1, 2],
            "self_correction_count": [0, 2],
            "word_finding_difficulty_count": [1, 2],
            "topic_drift_score": [0.1, 0.25],
            "temporal_disorganization_score": [0.1, 0.25],
            "sentence_fragment_ratio": [0.1, 0.2],
            "information_density_score": [0.6, 0.8],
            "coherence_score": [0.65, 0.85],
        },
    },
    "1": {
        "description": "mild dementia style speech",
        "transcript_style": (
            "出現較明顯停頓、重複、自我修正與找詞困難；"
            "事件順序有些混亂，細節比正常少，但仍大致可理解。"
        ),
        "duration_sec_range": (30, 45),
        "biomarker_targets": {
            "hesitation_count": [4, 7],
            "repetition_count": [2, 4],
            "self_correction_count": [1, 3],
            "word_finding_difficulty_count": [2, 4],
            "topic_drift_score": [0.25, 0.45],
            "temporal_disorganization_score": [0.25, 0.45],
            "sentence_fragment_ratio": [0.2, 0.35],
            "information_density_score": [0.45, 0.65],
            "coherence_score": [0.45, 0.7],
        },
    },
    "2": {
        "description": "moderate dementia style speech",
        "transcript_style": (
            "敘述明顯不連貫，重複較多，常使用模糊詞如「那個東西」「那邊」，"
            "可能忘記剛剛說到哪裡，時間順序混亂並且有離題。"
        ),
        "duration_sec_range": (30, 50),
        "biomarker_targets": {
            "hesitation_count": [7, 11],
            "repetition_count": [4, 7],
            "self_correction_count": [2, 5],
            "word_finding_difficulty_count": [4, 7],
            "topic_drift_score": [0.45, 0.7],
            "temporal_disorganization_score": [0.45, 0.75],
            "sentence_fragment_ratio": [0.35, 0.55],
            "information_density_score": [0.25, 0.5],
            "coherence_score": [0.25, 0.5],
        },
    },
    "3": {
        "description": "severe dementia style speech",
        "transcript_style": (
            "語句非常片段，資訊密度很低，重複、停頓、模糊詞很多；"
            "敘述可能無法完整完成，但仍要像自然口語，不要變成亂碼。"
        ),
        "duration_sec_range": (20, 40),
        "biomarker_targets": {
            "hesitation_count": [10, 16],
            "repetition_count": [6, 10],
            "self_correction_count": [2, 6],
            "word_finding_difficulty_count": [6, 10],
            "topic_drift_score": [0.65, 0.9],
            "temporal_disorganization_score": [0.65, 0.9],
            "sentence_fragment_ratio": [0.55, 0.8],
            "information_density_score": [0.1, 0.35],
            "coherence_score": [0.1, 0.35],
        },
    },
}


# -----------------------------
# Speaker prompt handling
# -----------------------------

DEFAULT_SPEAKERS = [
    {
        "speaker_id": "speaker_f_01",
        "speaker_gender": "female",
        "speaker_age_group": "adult",
        "speaker_prompt_audio_filename": "speaker_f_01.wav",
        "speaker_prompt_text_transcription": "今天天氣不錯，我早上去市場買了一些青菜和水果。",
    },
    {
        "speaker_id": "speaker_m_01",
        "speaker_gender": "male",
        "speaker_age_group": "adult",
        "speaker_prompt_audio_filename": "speaker_m_01.wav",
        "speaker_prompt_text_transcription": "我今天搭公車去附近的診所，路上人有一點多。",
    },
    {
        "speaker_id": "speaker_f_02",
        "speaker_gender": "female",
        "speaker_age_group": "older_adult",
        "speaker_prompt_audio_filename": "speaker_f_02.wav",
        "speaker_prompt_text_transcription": "吃完早餐以後，我整理了一下客廳，然後準備出門買東西。",
    },
    {
        "speaker_id": "speaker_m_02",
        "speaker_gender": "male",
        "speaker_age_group": "older_adult",
        "speaker_prompt_audio_filename": "speaker_m_02.wav",
        "speaker_prompt_text_transcription": "今天下午我在家裡看電視，晚一點要去便利商店買牛奶。",
    },
]


def ensure_speaker_csv(path: Path) -> None:
    """Create a template speaker_prompts.csv if it does not exist."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_SPEAKERS[0].keys())
        writer.writeheader()
        writer.writerows(DEFAULT_SPEAKERS)

    print(f"[INFO] Created template speaker CSV: {path}")
    print("[INFO] Put your actual WAV files in speaker_prompts/ with matching filenames.")


def load_speakers(path: Path) -> List[Dict[str, str]]:
    ensure_speaker_csv(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("speaker_prompts.csv is empty.")
    required = {
        "speaker_id",
        "speaker_gender",
        "speaker_age_group",
        "speaker_prompt_audio_filename",
        "speaker_prompt_text_transcription",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"speaker_prompts.csv is missing columns: {missing}")
    return rows


# -----------------------------
# Ollama / Qwen generation
# -----------------------------

def call_ollama(
    prompt: str,
    model: str = "qwen2.5:14b",
    host: str = "http://localhost:11434",
    temperature: float = 0.8,
    top_p: float = 0.9,
    timeout_sec: int = 180,
) -> str:
    """Call Ollama generate API using Python standard library."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": 4096,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}. "
            f"Make sure `ollama serve` is running and model {model} is pulled. Error: {e}"
        )


def build_prompt(sample_id: str, cdr_label: str, topic: str) -> str:
    rules = CDR_RULES[cdr_label]
    target = rules["biomarker_targets"]
    duration_min, duration_max = rules["duration_sec_range"]

    return f"""
你是台灣華語語料生成助手。請產生一筆「模擬語音研究用」的台灣華語口語逐字稿。
注意：這是合成語料，不是真實病患資料。

任務：
- 產生一段台灣華語口語敘述。
- 使用繁體中文。
- 主題：{topic}
- CDR 標籤：{cdr_label}
- CDR 描述：{rules["description"]}
- 口語風格規則：{rules["transcript_style"]}
- 目標朗讀長度：約 {duration_min} 到 {duration_max} 秒。
- 不要直接提到「失智、阿茲海默、CDR、病人、診斷、記憶測驗」等醫學詞。
- 不要寫成醫療報告。
- 不要加入旁白、標題、引號、條列。
- 內容要像一般人在台灣用華語自然講生活經驗。
- {TAIWAN_MANDARIN_HINTS[0]}
- {TAIWAN_MANDARIN_HINTS[1]}
- {TAIWAN_MANDARIN_HINTS[2]}
- {TAIWAN_MANDARIN_HINTS[3]}

請同時輸出語言生物標記。數值要符合 CDR 嚴重程度，並大致落在以下範圍：
{json.dumps(target, ensure_ascii=False, indent=2)}

你必須只輸出一個合法 JSON，不要輸出 markdown，不要使用 ```。
JSON schema:
{{
  "sample_id": "{sample_id}",
  "cdr_label": "{cdr_label}",
  "topic": "{topic}",
  "transcript": "繁體中文台灣華語口語逐字稿",
  "biomarkers": {{
    "hesitation_count": 整數,
    "repetition_count": 整數,
    "self_correction_count": 整數,
    "word_finding_difficulty_count": 整數,
    "topic_drift_score": 0到1的小數,
    "temporal_disorganization_score": 0到1的小數,
    "sentence_fragment_ratio": 0到1的小數,
    "information_density_score": 0到1的小數,
    "coherence_score": 0到1的小數
  }},
  "biomarker_notes": "用英文簡短說明這段逐字稿的語言特徵"
}}
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from Qwen output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # fallback: find the first JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{cleaned[:500]}")
    return json.loads(match.group(0))


# -----------------------------
# Validation and post-processing
# -----------------------------

def chinese_char_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def estimate_duration_sec(text: str) -> int:
    """
    Rough estimate for Mandarin spoken duration.
    4.0 to 4.8 Chinese characters/sec is a practical rough range.
    Dementia-style transcripts with pauses are slower, so punctuation/fillers add time.
    """
    chars = chinese_char_count(text)
    pause_bonus = text.count("……") * 1.0 + text.count("，") * 0.25 + text.count("。") * 0.4
    return max(5, round(chars / 4.3 + pause_bonus))


def has_banned_content(text: str) -> bool:
    return any(word in text for word in BANNED_CONTENT)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_biomarkers(cdr_label: str, biomarkers: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee all biomarker fields exist and are within valid type/range."""
    targets = CDR_RULES[cdr_label]["biomarker_targets"]
    normalized = {}

    for key, bounds in targets.items():
        low, high = bounds
        raw = biomarkers.get(key, None)

        if raw is None:
            if isinstance(low, int) and isinstance(high, int):
                raw = random.randint(low, high)
            else:
                raw = round(random.uniform(float(low), float(high)), 2)

        if "count" in key:
            normalized[key] = int(round(float(raw)))
            normalized[key] = max(0, normalized[key])
        else:
            normalized[key] = round(clamp(float(raw), 0.0, 1.0), 2)

    return normalized


def quality_check(row: Dict[str, Any]) -> List[str]:
    warnings = []
    transcript = row["transcript"]

    if has_banned_content(transcript):
        warnings.append("contains_banned_medical_terms")

    if chinese_char_count(transcript) < 40:
        warnings.append("too_short")

    if chinese_char_count(transcript) > 260:
        warnings.append("too_long")

    if re.search(r"[简体东门车后这为说]", transcript):
        # Not perfect, but catches common simplified characters.
        warnings.append("possible_simplified_chinese")

    return warnings


def dry_run_generate(sample_id: str, cdr_label: str, topic: str) -> Dict[str, Any]:
    """Fallback generator for testing pipeline without Ollama. Use only to test CSV creation."""
    examples = {
        "0": "今天早上我大概七點起床，先刷牙洗臉，然後去早餐店買蛋餅和豆漿。吃完以後我搭公車去菜市場，買了青菜、豆腐和一點水果，回家後就開始準備午餐。",
        "0.5": "今天早上我起來以後，嗯……先去刷牙，還是先去客廳，我想一下。後來我去早餐店買東西，買了蛋餅，還有那個，豆漿。回家的時候我本來要去市場，走到門口才想起來袋子忘了拿。",
        "1": "昨天我好像有去市場，嗯……應該是早上去的。我本來要買青菜，然後又想說要買豆腐，可是走到那邊我忘記要買什麼。後來我又回去問家人，才想起來是要買晚餐的東西。",
        "2": "我早上，嗯……早上有出去，去那個地方，買東西。買菜，還是買飯，我有點忘記。那個袋子，袋子我一直找，後來又在桌上。然後我好像有搭公車，可是又好像沒有，反正就是去那邊。",
        "3": "嗯……我早上，早上……那個，出去。買，買那個東西。袋子，袋子不見，後來又有。我要說什麼，嗯……菜，還是飯。那個人很多，很多。回來，回來以後……我忘了。",
    }
    targets = CDR_RULES[cdr_label]["biomarker_targets"]
    biomarkers = {}
    for k, (low, high) in targets.items():
        if "count" in k:
            biomarkers[k] = random.randint(int(low), int(high))
        else:
            biomarkers[k] = round(random.uniform(float(low), float(high)), 2)

    return {
        "sample_id": sample_id,
        "cdr_label": cdr_label,
        "topic": topic,
        "transcript": examples[cdr_label],
        "biomarkers": biomarkers,
        "biomarker_notes": "Dry-run example only. Replace with Qwen output for final dataset.",
    }


def generate_one(
    sample_id: str,
    cdr_label: str,
    topic: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.dry_run:
        return dry_run_generate(sample_id, cdr_label, topic)

    prompt = build_prompt(sample_id, cdr_label, topic)
    raw = call_ollama(
        prompt=prompt,
        model=args.model,
        host=args.ollama_host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_sec=args.timeout_sec,
    )
    parsed = extract_json(raw)
    parsed["_raw_model_output"] = raw
    return parsed


def make_sample_id(cdr_label: str, index: int) -> str:
    safe_label = cdr_label.replace(".", "_")
    return f"tw_md_cdr{safe_label}_{index:04d}"


def build_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_csv = Path(args.speaker_csv)
    speakers = load_speakers(speaker_csv)

    rows = []
    log_path = out_dir / "generation_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_f:
        for cdr_label, count in CDR_DISTRIBUTION.items():
            print(f"[INFO] Generating CDR {cdr_label}: {count} samples")

            for i in range(1, count + 1):
                sample_id = make_sample_id(cdr_label, i)
                topic = random.choice(TOPICS)
                speaker = random.choice(speakers)

                parsed: Optional[Dict[str, Any]] = None
                last_error: Optional[str] = None

                for attempt in range(1, args.max_retries + 1):
                    try:
                        parsed = generate_one(sample_id, cdr_label, topic, args)
                        transcript = str(parsed.get("transcript", "")).strip()

                        if not transcript:
                            raise ValueError("Empty transcript.")

                        if has_banned_content(transcript):
                            raise ValueError("Transcript contains banned medical terms.")

                        break
                    except Exception as e:
                        last_error = str(e)
                        print(f"[WARN] {sample_id} attempt {attempt} failed: {last_error}")
                        time.sleep(args.retry_sleep_sec)

                if parsed is None:
                    print(f"[ERROR] Failed {sample_id}; using dry-run fallback.")
                    parsed = dry_run_generate(sample_id, cdr_label, topic)
                    parsed["error"] = last_error

                transcript = str(parsed.get("transcript", "")).strip()
                biomarkers = normalize_biomarkers(cdr_label, parsed.get("biomarkers", {}))
                duration_sec = estimate_duration_sec(transcript)

                output_audio_filename = f"{sample_id}.wav"

                row = {
                    "sample_id": sample_id,
                    "cdr_label": cdr_label,
                    "cdr_description": CDR_RULES[cdr_label]["description"],
                    "language": LANGUAGE,
                    "script": SCRIPT,
                    "topic": topic,

                    # Qwen output
                    "transcript": transcript,
                    "content_to_synthesize": transcript,
                    "target_duration_sec": duration_sec,
                    "chinese_char_count": chinese_char_count(transcript),

                    # Speaker prompt fields for BreezyVoice
                    "speaker_id": speaker["speaker_id"],
                    "speaker_gender": speaker["speaker_gender"],
                    "speaker_age_group": speaker["speaker_age_group"],
                    "speaker_prompt_audio_filename": speaker["speaker_prompt_audio_filename"],
                    "speaker_prompt_text_transcription": speaker["speaker_prompt_text_transcription"],

                    # BreezyVoice output filename
                    "output_audio_filename": output_audio_filename,

                    # Linguistic biomarkers
                    "hesitation_count": biomarkers["hesitation_count"],
                    "repetition_count": biomarkers["repetition_count"],
                    "self_correction_count": biomarkers["self_correction_count"],
                    "word_finding_difficulty_count": biomarkers["word_finding_difficulty_count"],
                    "topic_drift_score": biomarkers["topic_drift_score"],
                    "temporal_disorganization_score": biomarkers["temporal_disorganization_score"],
                    "sentence_fragment_ratio": biomarkers["sentence_fragment_ratio"],
                    "information_density_score": biomarkers["information_density_score"],
                    "coherence_score": biomarkers["coherence_score"],

                    # Useful later for acoustic processing
                    "target_pause_density": pause_density_from_cdr(cdr_label),
                    "target_speech_rate": speech_rate_from_cdr(cdr_label),
                    "biomarker_notes": str(parsed.get("biomarker_notes", "")).strip(),

                    # Readiness / QC
                    "breezyvoice_ready": "yes",
                    "quality_warnings": "|".join(quality_check({
                        "transcript": transcript,
                        "cdr_label": cdr_label,
                    })),
                }

                rows.append(row)

                log_f.write(json.dumps({
                    "sample_id": sample_id,
                    "cdr_label": cdr_label,
                    "topic": topic,
                    "parsed": parsed,
                }, ensure_ascii=False) + "\n")

                if len(rows) % 25 == 0:
                    print(f"[INFO] Generated {len(rows)} / {sum(CDR_DISTRIBUTION.values())}")

    random.shuffle(rows)
    return rows


def pause_density_from_cdr(cdr_label: str) -> str:
    return {
        "0": "very_low",
        "0.5": "low",
        "1": "medium",
        "2": "high",
        "3": "very_high",
    }[cdr_label]


def speech_rate_from_cdr(cdr_label: str) -> str:
    return {
        "0": "normal",
        "0.5": "slightly_slow",
        "1": "slow",
        "2": "very_slow",
        "3": "fragmented_slow",
    }[cdr_label]


def save_dataset_metadata(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to save.")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_breezyvoice_batch(rows: List[Dict[str, Any]], path: Path) -> None:
    """
    BreezyVoice batch CSV columns:
    speaker_prompt_audio_filename,
    speaker_prompt_text_transcription,
    content_to_synthesize,
    output_audio_filename
    """
    fieldnames = [
        "speaker_prompt_audio_filename",
        "speaker_prompt_text_transcription",
        "content_to_synthesize",
        "output_audio_filename",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "speaker_prompt_audio_filename": row["speaker_prompt_audio_filename"],
                "speaker_prompt_text_transcription": row["speaker_prompt_text_transcription"],
                "content_to_synthesize": row["content_to_synthesize"],
                "output_audio_filename": row["output_audio_filename"],
            })


def save_distribution_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    summary = {}
    for row in rows:
        label = row["cdr_label"]
        summary[label] = summary.get(label, 0) + 1

    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "total": len(rows),
            "distribution": summary,
            "expected_distribution": CDR_DISTRIBUTION,
            "language": LANGUAGE,
            "script": SCRIPT,
        }, f, ensure_ascii=False, indent=2)



def split_dataset_stratified(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stratified train/val/test split by CDR label.
    This keeps every split balanced across CDR 0, 0.5, 1, 2, and 3.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio}. "
            f"Current ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(row["cdr_label"], []).append(row)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for cdr_label, group in grouped.items():
        group_copy = group[:]
        rng.shuffle(group_copy)

        n = len(group_copy)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        # Safety repair if unusual ratios cause rounding issues.
        if n_test < 0:
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val

        train_rows = group_copy[:n_train]
        val_rows = group_copy[n_train:n_train + n_val]
        test_rows = group_copy[n_train + n_val:]

        for row in train_rows:
            row["split"] = "train"
        for row in val_rows:
            row["split"] = "val"
        for row in test_rows:
            row["split"] = "test"

        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        splits["test"].extend(test_rows)

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def summarize_split_distribution(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for split_name, split_rows in splits.items():
        label_counts: Dict[str, int] = {}
        for row in split_rows:
            label = row["cdr_label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        summary[split_name] = {
            "total": len(split_rows),
            "distribution": dict(sorted(label_counts.items(), key=lambda item: float(item[0]))),
        }

    summary["all"] = {
        "total": sum(len(split_rows) for split_rows in splits.values()),
        "expected_distribution": CDR_DISTRIBUTION,
    }

    return summary


def save_split_files(
    splits: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Save metadata and BreezyVoice batch CSV for each split."""
    for split_name, split_rows in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        save_dataset_metadata(
            split_rows,
            split_dir / f"dataset_metadata_{split_name}.csv",
        )
        save_breezyvoice_batch(
            split_rows,
            split_dir / f"breezyvoice_batch_{split_name}.csv",
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default="tw_mandarin_dementia_dataset")
    parser.add_argument("--speaker-csv", default="speaker_prompts/speaker_prompts.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Create test CSVs without calling Qwen/Ollama.")

    # Stratified split ratios.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[INFO] Starting Taiwanese Mandarin dementia transcript generation")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Distribution: {CDR_DISTRIBUTION}")

    rows = build_dataset(args)

    out_dir = Path(args.output_dir)
    metadata_path = out_dir / "dataset_metadata.csv"
    breezyvoice_path = out_dir / "breezyvoice_batch.csv"
    summary_path = out_dir / "distribution_summary.json"
    split_summary_path = out_dir / "split_summary.json"

    # Stratified train/val/test split by CDR label.
    splits = split_dataset_stratified(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_dataset_metadata(rows, metadata_path)
    save_breezyvoice_batch(rows, breezyvoice_path)
    save_distribution_summary(rows, summary_path)
    save_split_files(splits, out_dir)

    split_summary = summarize_split_distribution(splits)
    with split_summary_path.open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"Generated samples: {len(rows)}")
    print(f"Saved full metadata: {metadata_path}")
    print(f"Saved full BreezyVoice batch CSV: {breezyvoice_path}")
    print(f"Saved distribution summary: {summary_path}")
    print(f"Saved split summary: {split_summary_path}")
    print("\nSplit distribution:")
    for split_name, info in split_summary.items():
        if split_name == "all":
            continue
        print(f"  {split_name}: {info['total']} samples -> {info['distribution']}")
    print("\nNext step:")
    print("Use breezyvoice_batch.csv for all audio, or use train/val/test BreezyVoice CSVs separately.")
    print("Make sure speaker_prompt_audio_filename files exist in your BreezyVoice prompt-audio folder.")


if __name__ == "__main__":
    main()
