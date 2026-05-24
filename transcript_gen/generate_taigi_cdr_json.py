"""
Generate Taigi-dominant CDR speech transcripts with Qwen through Ollama.

This version generates JSON event scripts for later TTS/audio rendering.

Outputs a folder-based JSON + text dataset:
  data/transcripts/new_generated_taigi_json/<run_id>/
    metadata.jsonl
    manifest.csv
    cdr_0/*.txt
    cdr_0/*.events.json
    cdr_0_5/*.txt
    cdr_0_5/*.events.json
    cdr_1/*.txt
    cdr_1/*.events.json
    cdr_2/*.txt
    cdr_2/*.events.json
    cdr_3/*.txt
    cdr_3/*.events.json
    splits/train.jsonl
    splits/val.jsonl
    splits/test.jsonl

Usage:
  python generate_taigi_cdr_json.py --samples-per-level 10 --model qwen2.5:14b
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from transcript_parser import parse_transcript
except ImportError:
    try:
        from src.transcript_gen.transcript_parser import parse_transcript
    except ImportError:
        parse_transcript = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "transcripts" / "new_generated_taigi_json"

CDR_LEVELS = [0, 0.5, 1, 2, 3]
ALLOWED_EVENTS = {"speech", "pause", "silence", "sigh", "cough"}
ALLOWED_MARKERS = ["[停頓]", "[長停頓]", "[沉默]", "[嘆氣]", "[咳嗽]"]

SCENARIOS = {
    "market": "講述今仔日去菜市仔買菜、和攤販講話、買菜轉去厝內。",
    "family": "講述囝仔和孫仔轉來厝內食飯，大家坐佇灶跤邊講話。",
    "morning": "講述早起後洗面、食飯、整理物件、準備出門。",
    "clinic": "在診間回答醫師問今日日期、早餐、家裡發生啥代誌。",
    "memory": "回想少年時陣的生活，例如讀冊、農忙、厝內長輩。",
}

CDR_STYLE = {
    0: {
        "label": "normal",
        "taigi_ratio": "60-70%",
        "length": "150-210字",
        "features": "語意清楚、順序完整，只有自然口語助詞，少量自然停頓。",
    },
    0.5: {
        "label": "very_mild",
        "taigi_ratio": "65-75%",
        "length": "130-190字",
        "features": "偶爾想袂起詞，會用那個、嗯、啊補位，但大致能回到主題。",
    },
    1: {
        "label": "mild",
        "taigi_ratio": "70-85%",
        "length": "110-180字",
        "features": "輕度失智；明顯找詞困難、重複、停頓、輕微離題，但仍可理解。",
    },
    2: {
        "label": "moderate",
        "taigi_ratio": "80-90%",
        "length": "70-120字",
        "features": (
            "中度失智；句子明顯破碎，常說一半停掉；重複同一詞或同一句；"
            "話題可從原情境聯想到家人、早餐、童年或身體不舒服，但不要完全亂跳。"
        ),
    },
    3: {
        "label": "severe",
        "taigi_ratio": "90-95%",
        "length": "25-65字",
        "features": (
            "重度失智；不可寫成完整敘事。只用很短、破碎、不完整的片段；"
            "頻繁中斷、沉默、重複字詞；突然切換到阿母、食飯、囝仔、天氣等熟悉主題。"
        ),
    },
}

# Machine-readable targets. These are used in the prompt and in validation.
CDR_PROFILE = {
    0: {
        "pause_count": (0, 1),
        "pause_range_ms": (200, 700),
        "word_finding_count": (0, 0),
        "repetition_count": (0, 0),
        "repair_count": (0, 0),
        "semantic_drift_count": (0, 0),
        "speech_rate": 1.00,
    },
    0.5: {
        "pause_count": (1, 3),
        "pause_range_ms": (500, 1200),
        "word_finding_count": (1, 2),
        "repetition_count": (0, 1),
        "repair_count": (0, 1),
        "semantic_drift_count": (0, 1),
        "speech_rate": 0.92,
    },
    1: {
        "pause_count": (3, 6),
        "pause_range_ms": (800, 2500),
        "word_finding_count": (2, 5),
        "repetition_count": (1, 3),
        "repair_count": (1, 3),
        "semantic_drift_count": (1, 2),
        "speech_rate": 0.85,
    },
    2: {
        "pause_count": (5, 9),
        "pause_range_ms": (1500, 4500),
        "word_finding_count": (4, 8),
        "repetition_count": (2, 5),
        "repair_count": (1, 4),
        "semantic_drift_count": (2, 4),
        "speech_rate": 0.75,
    },
    3: {
        "pause_count": (5, 12),
        "pause_range_ms": (2500, 7000),
        "word_finding_count": (3, 8),
        "repetition_count": (3, 8),
        "repair_count": (0, 3),
        "semantic_drift_count": (3, 6),
        "speech_rate": 0.65,
    },
}

CDR_RULES = {
    0: """CDR 0 專用規則：
- 正常認知長輩，不能有失智症狀。
- 內容要完整、有順序、能回答情境。
- 可有少量自然口語詞和最多1次短暫停頓。
- 不要大量重複、不要明顯離題、不要長停頓或沉默。""",
    0.5: """CDR 0.5 專用規則：
- 大致清楚完整，但偶爾找不到詞。
- 可出現1-2次「那個...」「想袂起來」。
- 可以短暫離題，但要自己拉回主題。
- 停頓少量，不要嚴重破碎。""",
    1: """CDR 1 專用規則：
- 輕度失智，仍能理解大意。
- 明顯找詞困難、重複、停頓、輕微離題。
- 可出現「彼個...彼個叫啥」「想袂起來」。
- 句子變短，但不要像CDR 3那樣只剩片段。""",
    2: """CDR 2 專用規則：
- 中度失智，句子明顯破碎。
- 每個 speech 片段盡量短，約8-14字。
- 要有3次以上中斷或重複。
- 至少2次話題聯想式跳走，例如從情境跳到早餐、家人、身體、以前的事。
- 多用長停頓或沉默事件。""",
    3: """CDR 3 專用規則：
- 重度失智，不要寫完整句子或完整故事。
- 每個 speech 片段約2-8字。
- 至少3次長停頓或沉默事件。
- 必須突然換話題，例如阿母、食飯、囝仔、天氣。
- 結尾可以未完成。

CDR 3 風格例子，只學形式，不要照抄：
「菜市仔... 阿母咧？嗯... 食飯... 囝仔，彼個... 冷啦... 袂記得...」""",
}

SYSTEM_PROMPT = """你是台灣本土語言、台語口語轉寫、臨床語言學專家。
請產生台灣長輩自然講話的逐字稿，用正體中文與台語漢字書寫。

重要規則：
- 內容必須台語/台灣閩南語（Taigi）為主，台式華語自然混入。
- 不要使用羅馬字、拼音、POJ、台羅。
- 不要使用簡體字；所有華語和台語漢字都要用正體中文。
- 不要加標題、說明、編號、引號。
- 不要使用「台語：」或任何語言標籤。
- 口吻要像台灣65-85歲長輩日常聊天，不要像書面作文。
- 常用台語詞可包含：今仔日、菜市仔、欲、袂、毋知、啥物、按呢、佇、伊、咧、攏、閣、足、誠、阿母、阿爸、囝仔、孫仔、厝、灶跤、食飯、轉去。
- 你必須只輸出有效 JSON，不要輸出 JSON 以外的任何文字。
"""


def cdr_dir_name(cdr: float) -> str:
    return f"cdr_{str(cdr).replace('.', '_')}"


def profile_as_prompt(cdr: float) -> str:
    profile = CDR_PROFILE[cdr]
    lines = []
    for key, value in profile.items():
        if isinstance(value, tuple):
            lines.append(f"- {key}: {value[0]} 到 {value[1]}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_prompt(cdr: float, scenario_key: str) -> str:
    style = CDR_STYLE[cdr]
    scenario = SCENARIOS[scenario_key]
    return f"""請產生一段台灣長輩口語逐字稿，並同時產生可供 TTS 合成的事件腳本。

CDR等級：{cdr}（{style["label"]}）
語言比例：台語/Taigi 約 {style["taigi_ratio"]}，其餘為自然台式華語。
情境：{scenario}
認知語言特徵：{style["features"]}
長度：{style["length"]}

{CDR_RULES[cdr]}

本 CDR 的量化目標：
{profile_as_prompt(cdr)}

共同要求：
- 可使用 嗯、啊、那個、就是、齁、啦、咧 等口語填充。
- 不要輸出英文、底線、代碼、Markdown、標題或註解。
- spoken_transcript 內可以包含 [停頓]、[長停頓]、[沉默]、[嘆氣]、[咳嗽]。
- event_script 裡的 speech.text 不可以包含任何方括號標記。
- event_script 的 type 只能是 speech、pause、silence、sigh、cough。
- pause、silence、sigh、cough 必須有 duration_ms。
- duration_ms 要符合 CDR 程度，不要全部一樣。
- 話題跳轉要像老人家自然聯想，不要隨機亂跳。

只輸出有效 JSON，格式必須完全如下：
{{
  "spoken_transcript": "逐字稿正文",
  "event_script": [
    {{"type": "speech", "text": "語音片段"}},
    {{"type": "pause", "duration_ms": 900}},
    {{"type": "speech", "text": "語音片段"}}
  ],
  "impairment_labels": {{
    "word_finding_count": 0,
    "repetition_count": 0,
    "repair_count": 0,
    "semantic_drift_count": 0,
    "orientation_error": false,
    "code_switch_count": 0
  }}
}}"""


def clean_generated_text(text: str, preserve_markers: bool = True) -> str:
    """Remove common model artifacts. Optionally preserve approved event markers."""
    replacements = {
        "那个": "那個",
        "什么": "什麼",
        "怎么": "怎麼",
        "说": "說",
        "来": "來",
        "着": "著",
        "手机": "手機",
        "买": "買",
        "猪": "豬",
        "汤": "湯",
        "炉": "爐",
        "裏": "裡",
        "嗯知": "毋知",
        "(嘆氣)": "[嘆氣]",
        "（嘆氣）": "[嘆氣]",
        "(咳嗽)": "[咳嗽]",
        "（咳嗽）": "[咳嗽]",
        "_RCCS": "",
        "RCCS": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    if preserve_markers:
        placeholders = {f"§{index}§": marker for index, marker in enumerate(ALLOWED_MARKERS)}
        for placeholder, marker in placeholders.items():
            text = text.replace(marker, placeholder)

        text = re.sub(r"長停頓[.。…]*", "[長停頓]", text)
        text = re.sub(r"停頓[.。…]*", "[停頓]", text)
        text = re.sub(r"沉默[.。…]*", "[沉默]", text)
        for placeholder, marker in placeholders.items():
            text = text.replace(marker, placeholder)
        text = re.sub(r"\[(?!停頓\]|長停頓\]|沉默\]|嘆氣\]|咳嗽\])[^]]+\]", "", text)
        text = text.replace("[", "").replace("]", "")
        for placeholder, marker in placeholders.items():
            text = text.replace(placeholder, marker)
    else:
        text = re.sub(r"\[(停頓|長停頓|沉默|嘆氣|咳嗽)\]", "", text)
        text = re.sub(r"\[[^]]+\]", "", text)

    text = re.sub(r"[A-Za-z_]+", "", text)
    text = re.sub(r"\s+", " ", text)
    lines = [line.strip(" 「」\"'") for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model response."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")
        return json.loads(match.group(0))


def marker_to_event(marker: str, cdr: float) -> dict[str, Any]:
    profile = CDR_PROFILE[cdr]
    low, high = profile["pause_range_ms"]
    if marker == "[停頓]":
        return {"type": "pause", "duration_ms": random.randint(max(200, low), min(1200, high))}
    if marker == "[長停頓]":
        return {"type": "pause", "duration_ms": random.randint(max(1200, low), high)}
    if marker == "[沉默]":
        return {"type": "silence", "duration_ms": random.randint(max(1500, low), high)}
    if marker == "[嘆氣]":
        return {"type": "sigh", "duration_ms": random.randint(600, 1400)}
    if marker == "[咳嗽]":
        return {"type": "cough", "duration_ms": random.randint(400, 1200)}
    return {"type": "pause", "duration_ms": random.randint(low, high)}


def events_from_marked_transcript(text: str, cdr: float) -> list[dict[str, Any]]:
    """Fallback: convert a marked transcript into an event_script."""
    pattern = r"(\[停頓\]|\[長停頓\]|\[沉默\]|\[嘆氣\]|\[咳嗽\])"
    parts = re.split(pattern, text)
    events: list[dict[str, Any]] = []
    for part in parts:
        if not part:
            continue
        if part in ALLOWED_MARKERS:
            events.append(marker_to_event(part, cdr))
        else:
            speech = clean_generated_text(part, preserve_markers=False)
            if speech:
                events.append({"type": "speech", "text": speech})
    return events


def normalize_event_script(events: Any, cdr: float) -> list[dict[str, Any]]:
    """Validate/clean event script returned by the LLM."""
    if not isinstance(events, list):
        return []

    profile = CDR_PROFILE[cdr]
    low, high = profile["pause_range_ms"]
    normalized: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).strip().lower()
        if event_type not in ALLOWED_EVENTS:
            continue

        if event_type == "speech":
            text = clean_generated_text(str(event.get("text", "")), preserve_markers=False)
            if text:
                normalized.append({"type": "speech", "text": text})
            continue

        try:
            duration_ms = int(float(event.get("duration_ms", random.randint(low, high))))
        except (TypeError, ValueError):
            duration_ms = random.randint(low, high)

        if event_type in {"pause", "silence"}:
            duration_ms = max(200, min(duration_ms, 8000))
        elif event_type == "sigh":
            duration_ms = max(300, min(duration_ms, 2500))
        elif event_type == "cough":
            duration_ms = max(250, min(duration_ms, 2000))

        normalized.append({"type": event_type, "duration_ms": duration_ms})

    # Merge adjacent speech events for cleaner TTS chunks.
    merged: list[dict[str, Any]] = []
    for event in normalized:
        if event["type"] == "speech" and merged and merged[-1]["type"] == "speech":
            merged[-1]["text"] = f'{merged[-1]["text"]} {event["text"]}'.strip()
        else:
            merged.append(event)

    return merged


def transcript_from_events(events: list[dict[str, Any]]) -> str:
    """Create readable transcript from normalized event_script."""
    pieces = []
    for event in events:
        event_type = event["type"]
        if event_type == "speech":
            pieces.append(event["text"])
        elif event_type == "pause":
            duration = int(event.get("duration_ms", 0))
            pieces.append("[長停頓]" if duration >= 1500 else "[停頓]")
        elif event_type == "silence":
            pieces.append("[沉默]")
        elif event_type == "sigh":
            pieces.append("[嘆氣]")
        elif event_type == "cough":
            pieces.append("[咳嗽]")
    return " ".join(pieces).strip()


def default_labels() -> dict[str, Any]:
    return {
        "word_finding_count": 0,
        "repetition_count": 0,
        "repair_count": 0,
        "semantic_drift_count": 0,
        "orientation_error": False,
        "code_switch_count": 0,
    }


def normalize_labels(labels: Any) -> dict[str, Any]:
    result = default_labels()
    if not isinstance(labels, dict):
        return result
    for key in result:
        value = labels.get(key, result[key])
        if key == "orientation_error":
            result[key] = bool(value)
        else:
            try:
                result[key] = max(0, int(value))
            except (TypeError, ValueError):
                result[key] = 0
    return result


def event_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    pause_events = [e for e in events if e["type"] in {"pause", "silence"}]
    speech_events = [e for e in events if e["type"] == "speech"]
    pause_total_ms = sum(int(e.get("duration_ms", 0)) for e in pause_events)
    speech_chars = sum(len(e.get("text", "")) for e in speech_events)
    return {
        "event_count": len(events),
        "speech_event_count": len(speech_events),
        "pause_event_count": len(pause_events),
        "sigh_count": sum(1 for e in events if e["type"] == "sigh"),
        "cough_count": sum(1 for e in events if e["type"] == "cough"),
        "pause_total_ms": pause_total_ms,
        "mean_pause_ms": round(pause_total_ms / len(pause_events), 2) if pause_events else 0,
        "speech_char_count": speech_chars,
    }


def validate_sample(cdr: float, text: str, events: list[dict[str, Any]]) -> tuple[bool, str]:
    """Simple guardrails to reject clearly mismatched generations."""
    if not text or len(text) < 10:
        return False, "text too short"
    if not events:
        return False, "empty event_script"
    if not any(event["type"] == "speech" for event in events):
        return False, "no speech events"

    stats = event_stats(events)
    pause_count = int(stats["pause_event_count"])
    speech_count = int(stats["speech_event_count"])

    min_pauses, max_pauses = CDR_PROFILE[cdr]["pause_count"]
    # Be slightly permissive because local LLMs can vary.
    if pause_count < max(0, min_pauses - 1):
        return False, f"too few pauses for CDR {cdr}: {pause_count}"
    if cdr == 0 and pause_count > 2:
        return False, f"too many pauses for CDR 0: {pause_count}"
    if cdr == 3 and speech_count > 12:
        return False, f"too many speech chunks for CDR 3: {speech_count}"
    if cdr <= 1 and "[沉默]" in text and cdr < 1:
        return False, f"silence marker too severe for CDR {cdr}"

    return True, "ok"


def parse_model_output(raw_response: str, cdr: float) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Parse LLM JSON. Falls back to marked-transcript parsing if JSON repair fails."""
    try:
        obj = extract_json_object(raw_response)
        spoken_text = clean_generated_text(str(obj.get("spoken_transcript", "")), preserve_markers=True)
        events = normalize_event_script(obj.get("event_script", []), cdr)
        labels = normalize_labels(obj.get("impairment_labels", {}))
        if not events and spoken_text:
            events = events_from_marked_transcript(spoken_text, cdr)
        if not spoken_text and events:
            spoken_text = transcript_from_events(events)
        repaired = False
    except Exception:
        spoken_text = clean_generated_text(raw_response, preserve_markers=True)
        events = events_from_marked_transcript(spoken_text, cdr)
        labels = default_labels()
        repaired = True

    # Ensure transcript and events agree after normalization.
    if events:
        spoken_text = transcript_from_events(events)

    return spoken_text, events, labels, {"json_repaired_or_fallback": repaired}


def ollama_chat(base_url: str, model: str, system: str, user: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 1300,
        },
    }
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def verify_ollama(base_url: str, model: str) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    models = [item["name"] for item in response.json().get("models", [])]
    if model not in models:
        raise RuntimeError(f"Model {model!r} not found. Available models: {', '.join(models) or 'none'}")


def load_existing_records(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """Load records already written by an interrupted/resumed run."""
    if not metadata_path.exists():
        return {}

    records = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["sample_id"]] = record
    return records


def write_dataset_outputs(output_dir: Path, records: list[dict[str, Any]], seed: int) -> None:
    """Write full metadata plus stratified train/val/test split manifests."""
    records = sorted(records, key=lambda item: (float(item["cdr_level"]), item["sample_id"]))

    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample_id",
                "cdr_level",
                "cdr_label",
                "scenario",
                "text_path",
                "events_path",
                "generation_time_sec",
                "validation_status",
                "raw_response",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    by_cdr: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        by_cdr.setdefault(float(record["cdr_level"]), []).append(record)

    split_records = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for cdr in CDR_LEVELS:
        items = list(by_cdr.get(float(cdr), []))
        rng.shuffle(items)
        count = len(items)
        train_end = int(count * 0.70)
        val_end = train_end + int(count * 0.15)
        split_records["train"].extend(items[:train_end])
        split_records["val"].extend(items[train_end:val_end])
        split_records["test"].extend(items[val_end:])

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in split_records.items():
        split_items = sorted(split_items, key=lambda item: (float(item["cdr_level"]), item["sample_id"]))
        (split_dir / f"{split_name}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in split_items),
            encoding="utf-8",
        )
        with (split_dir / f"{split_name}.txt").open("w", encoding="utf-8") as fp:
            for record in split_items:
                fp.write(record["text_path"] + "\n")


def generate_one_sample(
    args: argparse.Namespace,
    cdr: float,
    scenario: str,
    sample_id: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], str, dict[str, Any], str]:
    prompt = build_prompt(cdr, scenario)
    last_reason = "not generated"

    for attempt in range(1, args.max_retries + 2):
        raw_response = ollama_chat(args.url, args.model, SYSTEM_PROMPT, prompt, args.temperature)
        text, events, labels, parse_info = parse_model_output(raw_response, cdr)
        is_valid, reason = validate_sample(cdr, text, events)
        if is_valid:
            parse_info["attempt"] = attempt
            return text, events, labels, raw_response, parse_info, "ok"

        last_reason = reason
        prompt = (
            build_prompt(cdr, scenario)
            + f"\n\n上一版不合格原因：{reason}。請重新產生，務必符合 CDR {cdr} 的量化目標。"
        )

    # Last-resort return: keep data but mark failed validation so you can audit it.
    parse_info["attempt"] = args.max_retries + 1
    parse_info["last_validation_error"] = last_reason
    return text, events, labels, raw_response, parse_info, f"failed_validation: {last_reason}"


def generate(args: argparse.Namespace) -> Path:
    verify_ollama(args.url, args.model)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    random.seed(args.seed)
    scenarios = list(SCENARIOS.keys())
    records_by_id = load_existing_records(metadata_path)
    total = args.samples_per_level * len(CDR_LEVELS)

    if records_by_id:
        print(f"resuming {run_id}: {len(records_by_id)}/{total} records already exist")

    with metadata_path.open("a", encoding="utf-8") as meta_fp:
        for cdr in CDR_LEVELS:
            level_dir = output_dir / cdr_dir_name(cdr)
            level_dir.mkdir(parents=True, exist_ok=True)

            for index in range(args.samples_per_level):
                scenario = scenarios[(index + CDR_LEVELS.index(cdr)) % len(scenarios)]
                sample_id = f"{cdr_dir_name(cdr)}_{scenario}_{index:04d}"
                if sample_id in records_by_id:
                    continue

                start = time.time()
                text, events, labels, raw_response, parse_info, validation_status = generate_one_sample(
                    args=args,
                    cdr=cdr,
                    scenario=scenario,
                    sample_id=sample_id,
                )
                elapsed = round(time.time() - start, 2)

                if not text:
                    print(f"empty response: {sample_id}", file=sys.stderr)
                    continue

                text_path = level_dir / f"{sample_id}.txt"
                events_path = level_dir / f"{sample_id}.events.json"

                text_path.write_text(text + "\n", encoding="utf-8")
                events_path.write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "cdr_level": cdr,
                            "cdr_label": CDR_STYLE[cdr]["label"],
                            "scenario": scenario,
                            "spoken_transcript": text,
                            "event_script": events,
                            "impairment_labels": labels,
                            "event_stats": event_stats(events),
                            "speech_rate_target": CDR_PROFILE[cdr]["speech_rate"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                if parse_transcript is not None:
                    parsed = parse_transcript(
                        raw_text=text,
                        transcript_id=sample_id,
                        cdr_level=cdr,
                        scenario=scenario,
                    ).to_dict()
                else:
                    parsed = {}

                record = {
                    "sample_id": sample_id,
                    "cdr_level": cdr,
                    "cdr_label": CDR_STYLE[cdr]["label"],
                    "scenario": scenario,
                    "model": args.model,
                    "temperature": args.temperature,
                    "text_path": str(text_path.relative_to(PROJECT_ROOT)),
                    "events_path": str(events_path.relative_to(PROJECT_ROOT)),
                    "generation_time_sec": elapsed,
                    "validation_status": validation_status,
                    "raw_text": text,
                    "raw_response": raw_response,
                    "event_script": events,
                    "event_stats": event_stats(events),
                    "impairment_labels": labels,
                    "speech_rate_target": CDR_PROFILE[cdr]["speech_rate"],
                    "parse_info": parse_info,
                    "parsed": parsed,
                }
                meta_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_fp.flush()
                records_by_id[sample_id] = record

                stats = record["event_stats"]
                print(
                    f"{len(records_by_id)}/{total} | {sample_id} | {elapsed}s | "
                    f"{len(text)} chars | pauses={stats['pause_event_count']} | {validation_status}"
                )

    write_dataset_outputs(output_dir, list(records_by_id.values()), args.seed)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Taigi-dominant JSON event transcripts for all CDR levels."
    )
    parser.add_argument("--samples-per-level", type=int, default=1)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count when the generated sample fails validation.",
    )
    args = parser.parse_args()

    output_dir = generate(args)
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
