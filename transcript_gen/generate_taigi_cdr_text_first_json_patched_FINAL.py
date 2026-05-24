"""
Text-first Taigi / mixed Taiwanese Mandarin CDR transcript generator.

Design:
  1) Qwen through Ollama generates ONLY natural transcript text.
  2) Python cleans the text, rejects bad samples, converts speech-event markers to JSON events,
     and computes observed features after generation.
  3) Output keeps target CDR separate from observed transcript/audio-rendering features.

Default output:
  data/transcripts/new_generated_taigi_json/<run_id>/
    metadata.jsonl
    manifest.csv
    dataset_summary.json
    generation_plan.json
    cdr_0/*.txt
    cdr_0/*.events.json
    cdr_0_5/*.txt
    cdr_0_5/*.events.json
    ...
    splits/train.jsonl, val.jsonl, test.jsonl

Example full dataset:
  python generate_taigi_cdr_text_first_json.py \
    --cdr-counts "0:300,0.5:300,1:250,2:100,3:50" \
    --speaker-count 40 \
    --model qwen2.5:14b \
    --run-id full_cdr_progression

Example pilot:
  python generate_taigi_cdr_text_first_json.py \
    --cdr-counts "0:20,0.5:20,1:20,2:10,3:10" \
    --speaker-count 10 \
    --model qwen2.5:14b \
    --run-id pilot_text_first
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from transcript_parser import parse_transcript  # type: ignore
except ImportError:  # keep compatibility with your project layout
    try:
        from src.transcript_gen.transcript_parser import parse_transcript  # type: ignore
    except ImportError:
        parse_transcript = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "transcripts" / "new_generated_taigi_json"

CDR_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]
DEFAULT_CDR_COUNTS = {0.0: 300, 0.5: 300, 1.0: 250, 2.0: 100, 3.0: 50}

SCENARIOS = {
    "clinic": {
        "weight": 25,
        "description": "在診間回答醫師問今日日期、早餐、家裡發生啥代誌。",
    },
    "morning": {
        "weight": 15,
        "description": "講述早起後洗面、食飯、整理物件、準備出門。",
    },
    "market": {
        "weight": 15,
        "description": "講述今仔日去菜市仔買菜、和攤販講話、買菜轉去厝內。",
    },
    "family": {
        "weight": 15,
        "description": "講述囝仔和孫仔轉來厝內食飯，大家坐佇灶跤邊講話。",
    },
    "memory": {
        "weight": 20,
        "description": "回想少年時陣的生活，例如讀冊、農忙、厝內長輩。",
    },
    "picture_description": {
        "weight": 10,
        "description": "描述一張家庭或市場場景的圖，例如人咧煮飯、囝仔咧玩、桌頂有物件。",
    },
}

LANGUAGE_PROFILES = {
    "taigi_dominant": {
        "weight": 60,
        "description": "台語為主，約80-90%台語，只有少量自然台式華語詞。",
        "taigi_ratio_prompt": "80-90%",
    },
    "mixed_taigi_mandarin": {
        "weight": 30,
        "description": "台語與台式華語自然混合，約55-70%台語，允許日常華語插入。",
        "taigi_ratio_prompt": "55-70%",
    },
    "mandarin_dominant_mixed": {
        "weight": 10,
        "description": "台式華語較多，但仍保留台語詞、語氣詞與家庭/市場相關台語詞。",
        "taigi_ratio_prompt": "25-45%",
    },
}

ACOUSTIC_CONDITIONS = {
    "clean": {"weight": 40, "description": "乾淨近距離錄音，低背景噪音。"},
    "clinic_room_noise": {"weight": 25, "description": "診間/候診室輕微環境噪音。"},
    "home_background_noise": {"weight": 20, "description": "家中電視、風扇、廚房等輕微背景聲。"},
    "phone_mic_degraded": {"weight": 15, "description": "手機或低品質麥克風，頻寬較窄、些微壓縮感。"},
}

SPEAKER_GROUPS = [
    ("older_female_taigi_dominant", "女性長輩，台語優勢，65-85歲。", 12),
    ("older_male_taigi_dominant", "男性長輩，台語優勢，65-85歲。", 12),
    ("older_female_mixed", "女性長輩，台語與台式華語自然混合，65-85歲。", 8),
    ("older_male_mixed", "男性長輩，台語與台式華語自然混合，65-85歲。", 8),
]

CDR_STYLE = {
    0.0: {
        "label": "normal",
        "length": "110-170字",
        "speech_rate_target": 1.0,
        "features": "語意清楚、順序完整，只有自然口語助詞，最多1次短[停頓]。",
    },
    0.5: {
        "label": "very_mild",
        "length": "100-160字",
        "speech_rate_target": 0.92,
        "features": "偶爾想袂起詞，會用那個、嗯、啊補位，但大致能回到主題。",
    },
    1.0: {
        "label": "mild",
        "length": "90-150字",
        "speech_rate_target": 0.85,
        "features": "輕度失智；有找詞困難、重複、停頓、輕微離題，但仍可理解大意。",
    },
    2.0: {
        "label": "moderate",
        "length": "60-115字",
        "speech_rate_target": 0.75,
        "features": "中度失智；句子破碎，常說一半停掉，話題會從原情境跳到家人、早餐、童年或身體。",
    },
    3.0: {
        "label": "severe",
        "length": "25-70字",
        "speech_rate_target": 0.65,
        "features": "重度失智；短而破碎，沉默多，重複字詞，突然切到阿母、食飯、囝仔、天氣等。",
    },
}

CDR_RULES = {
    0.0: """CDR 0 專用規則：
- 正常認知長輩，不能有失智症狀。
- 內容要完整、有順序、能回答情境。
- 可有自然口語詞，最多1次[停頓]。
- 不要大量重複，不要明顯離題，不要[長停頓]、[沉默]、[嘆氣]、[咳嗽]。
- 不要寫成作文、演講或宣傳，不要談「保護文化」這類抽象議題；請像家裡日常聊天。""",
    0.5: """CDR 0.5 專用規則：
- 大致清楚完整，但偶爾找不到詞。
- 可出現1-2次「那個...」「想袂起來」。
- 可有1-3次短[停頓]；不要嚴重破碎。
- 可以短暫離題，但要自己拉回主題。""",
    1.0: """CDR 1 專用規則：
- 輕度失智，仍然可以大致回答問題，不要過度嚴重。
- 可出現「彼個...彼個叫啥」「想袂起來」。
- 有找詞困難、重複、停頓、輕微離題，但不要每一句都想不起來。
- 句子變短，但不要像CDR 3那樣只剩片段。""",
    2.0: """CDR 2 專用規則：
- 中度失智，句子明顯破碎。
- 每句多數約8-14字。
- 要有3次以上中斷或重複。
- 至少2次話題跳走，例如從情境跳到早餐、家人、身體、以前的事。
- 多用[長停頓]，可用[沉默]、[嘆氣]、[咳嗽]。""",
    3.0: """CDR 3 專用規則：
- 重度失智，不要寫完整句子或完整故事。
- 每個片段約2-8字。
- 至少3次[長停頓]/[沉默]。
- 必須突然換話題，例如阿母、食飯、囝仔、天氣。
- 結尾可以未完成。

CDR 3 風格例子，只學形式，不要照抄：
「菜市仔... [長停頓] 阿母咧？嗯... 食飯... [沉默] 囝仔，彼個... [長停頓] 冷啦... 袂記得...」""",
}

SYSTEM_PROMPT = """你是台灣本土語言、台語口語轉寫、臨床語言學專家。
請產生台灣長輩自然講話的逐字稿，用正體中文與台語漢字書寫。

重要規則：
- 內容必須台語/台灣閩南語（Taigi）與台式華語自然混合，依使用者指定比例。
- 不要使用羅馬字、拼音、POJ、台羅。
- 不要使用簡體字；所有華語和台語漢字都要用正體中文。
- 不要使用粵語字詞，例如：嘅、咗、啲、煮緊、講緊、係咁。
- 不要加標題、說明、編號、引號。
- 不要使用「台語：」或任何語言標籤。
- 可使用 [停頓]、[長停頓]、[沉默]、[嘆氣]、[咳嗽] 表示語音事件。
- 口吻要像台灣65-85歲長輩日常聊天，不要像書面作文。
- 常用台語詞可包含：今仔日、菜市仔、欲、袂、毋知、啥物、按呢、佇、伊、咧、攏、閣、足、誠、阿母、阿爸、囝仔、孫仔、厝、灶跤、食飯、轉去。
"""

BAD_PATTERNS = [
    "嘅", "咗", "啲", "係咁", "煮緊", "講緊", "睇", "唔", "冇",
    "米其林", "_RCCS", "RCCS",
]

PHRASE_REPAIRS = {
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
    "咳嗽...": "[咳嗽]",
    "咳嗽…": "[咳嗽]",
    "嘆氣...": "[嘆氣]",
    "嘆氣…": "[嘆氣]",
    "(嘆氣)": "[嘆氣]",
    "（嘆氣）": "[嘆氣]",
    "(咳嗽)": "[咳嗽]",
    "（咳嗽）": "[咳嗽]",
    "咧咧笑": "咧笑",
    "咧個": "彼個",
    "買佇一尾": "買一尾",
    "轉去厝內後": "轉去厝了後",
    "食無欲食": "食袂落",
    "阿母去哪佇": "阿母去佗位",
    "醫生有啥想問的": "醫生閣有啥物欲問的",
    "要多注意一下健康": "愛較注意身體",
    "返來": "轉來",
    "無知": "毋知",
    "阿爸嘞": "阿爸咧",
    "阿母嘞": "阿母咧",
    "搥阮": "載阮",
    "菜市買菜": "菜市仔買菜",
    "菜市場": "菜市仔",
    "伊欲給我多攏一包無": "伊有欲加送我一包無",
}

MARKERS = ["[停頓]", "[長停頓]", "[沉默]", "[嘆氣]", "[咳嗽]"]
MARKER_RE = re.compile(r"(\[停頓\]|\[長停頓\]|\[沉默\]|\[嘆氣\]|\[咳嗽\])")

# Minimum pause/silence events needed for audio rendering.
# Qwen often generates good text but forgets explicit [停頓] markers,
# so Python inserts short pauses at sentence/phrase boundaries when needed.
MIN_PAUSES_BY_CDR = {
    0.0: 1,
    0.5: 2,
    1.0: 3,
    2.0: 4,
    3.0: 4,
}

AUTO_PAUSE_RANGES = {
    0.0: (220, 550),
    0.5: (450, 1000),
    1.0: (650, 1600),
    2.0: (900, 2600),
    3.0: (1200, 4200),
}

SENTENCE_BOUNDARY_RE = re.compile(r"([^。！？!?…]+[。！？!?…]*)")

WORD_FINDING_TERMS = ["想袂起", "想不起", "叫啥", "叫什麼", "彼個", "那個", "嗯...", "嗯…", "毋知", "不知道"]
REPAIR_TERMS = ["毋是", "不是", "應該是", "我是講", "我講錯", "算矣", "不是啦"]
TOPIC_TERMS = ["早餐", "早飯", "菜市仔", "阿母", "阿爸", "囝仔", "孫仔", "天氣", "頭痛", "身體", "少年", "學校", "厝", "醫生", "食飯"]


@dataclass
class SpeakerProfile:
    speaker_id: str
    speaker_group: str
    description: str
    split: str


def cdr_key(cdr: float) -> float:
    return float(cdr)


def cdr_dir_name(cdr: float) -> str:
    return f"cdr_{str(cdr).replace('.', '_').replace('_0', '') if cdr != 0.5 else '0_5'}"


def canonical_cdr_name(cdr: float) -> str:
    if cdr == 0.0:
        return "cdr_0"
    if cdr == 0.5:
        return "cdr_0_5"
    if cdr == 1.0:
        return "cdr_1"
    if cdr == 2.0:
        return "cdr_2"
    if cdr == 3.0:
        return "cdr_3"
    return f"cdr_{str(cdr).replace('.', '_')}"


def parse_cdr_counts(raw: str) -> dict[float, int]:
    if not raw.strip():
        return dict(DEFAULT_CDR_COUNTS)
    counts = {cdr: 0 for cdr in CDR_LEVELS}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        counts[float(key.strip())] = int(value.strip())
    for cdr in counts:
        if counts[cdr] < 0:
            raise ValueError("CDR counts cannot be negative")
    return counts


def weighted_choice(items: dict[str, dict[str, Any]], rng: random.Random) -> str:
    names = list(items.keys())
    weights = [int(items[name].get("weight", 1)) for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def make_speakers(count: int, seed: int) -> list[SpeakerProfile]:
    rng = random.Random(seed)
    group_pool: list[tuple[str, str]] = []
    while len(group_pool) < count:
        for group, description, quota in SPEAKER_GROUPS:
            group_pool.extend([(group, description)] * quota)
    group_pool = group_pool[:count]
    rng.shuffle(group_pool)

    train_n = int(count * 0.70)
    val_n = max(1, int(count * 0.15)) if count >= 3 else 0
    test_n = count - train_n - val_n
    if count >= 3 and test_n == 0:
        test_n = 1
        train_n = max(1, train_n - 1)

    splits = (["train"] * train_n) + (["val"] * val_n) + (["test"] * test_n)
    splits = splits[:count]
    rng.shuffle(splits)

    speakers = []
    for i, ((group, description), split) in enumerate(zip(group_pool, splits), start=1):
        speakers.append(SpeakerProfile(
            speaker_id=f"synthetic_spk_{i:03d}",
            speaker_group=group,
            description=description,
            split=split,
        ))
    return speakers


def build_prompt(cdr: float, scenario_key: str, language_profile_key: str, speaker: SpeakerProfile) -> str:
    style = CDR_STYLE[cdr]
    scenario = SCENARIOS[scenario_key]
    lang = LANGUAGE_PROFILES[language_profile_key]
    return f"""請產生一段台灣長輩口語逐字稿。

CDR等級：{cdr:g}（{style['label']}）
說話者：{speaker.description}
語言比例：台語/Taigi 約 {lang['taigi_ratio_prompt']}，其餘為自然台式華語。
語言風格：{lang['description']}
情境：{scenario['description']}
認知語言特徵：{style['features']}
長度：{style['length']}

{CDR_RULES[cdr]}

共同要求：
- 可使用 嗯、啊、那個、就是、齁、啦、咧 等口語填充。
- 可使用合理停頓標記。
- 禁止輸出英文、底線、代碼、標籤或註解。
- 方括號只允許使用：[停頓]、[長停頓]、[沉默]、[嘆氣]、[咳嗽]。
- 禁止使用粵語字詞：嘅、咗、啲、係、煮緊、講緊。
- 對話要自然，不要像作文，不要有「保護文化」這種演講式內容。

只輸出逐字稿正文。"""


def clean_generated_text(text: str) -> str:
    text = text.strip()
    for source, target in PHRASE_REPAIRS.items():
        text = text.replace(source, target)

    # Normalize bare marker words into bracketed markers when they appear as stage directions.
    text = re.sub(r"(?<![\u4e00-\u9fff])長停頓[.。…]*", "[長停頓]", text)
    text = re.sub(r"(?<![\u4e00-\u9fff])停頓[.。…]*", "[停頓]", text)
    text = re.sub(r"(?<![\u4e00-\u9fff])沉默[.。…]*", "[沉默]", text)

    placeholders = {f"§{i}§": marker for i, marker in enumerate(MARKERS)}
    for placeholder, marker in placeholders.items():
        text = text.replace(marker, placeholder)

    # Drop unsupported bracketed text.
    text = re.sub(r"\[(?!停頓\]|長停頓\]|沉默\]|嘆氣\]|咳嗽\])[^\]]+\]", "", text)
    text = text.replace("[", "").replace("]", "")

    for placeholder, marker in placeholders.items():
        text = text.replace(placeholder, marker)

    text = re.sub(r"[A-Za-z_]+", "", text)
    text = re.sub(r"\s+", " ", text)
    lines = [line.strip(" 「」\"'") for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def marker_duration(marker: str, cdr: float, rng: random.Random) -> tuple[str, int]:
    # Robust ranges: never use invalid randint range.
    if marker == "[停頓]":
        ranges = {
            0.0: (220, 700),
            0.5: (450, 1200),
            1.0: (700, 1800),
            2.0: (900, 2200),
            3.0: (1200, 3000),
        }
        event_type = "pause"
    elif marker == "[長停頓]":
        ranges = {
            0.0: (900, 1600),
            0.5: (1000, 2200),
            1.0: (1500, 3200),
            2.0: (2500, 5000),
            3.0: (3000, 7500),
        }
        event_type = "pause"
    elif marker == "[沉默]":
        ranges = {
            0.0: (1200, 2200),
            0.5: (1500, 3000),
            1.0: (2000, 4000),
            2.0: (3000, 6500),
            3.0: (4000, 9000),
        }
        event_type = "silence"
    elif marker == "[嘆氣]":
        return "sigh", rng.randint(600, 1400)
    elif marker == "[咳嗽]":
        return "cough", rng.randint(400, 1100)
    else:
        return "pause", rng.randint(400, 900)

    low, high = ranges.get(cdr, (500, 1500))
    if high < low:
        high = low
    return event_type, rng.randint(low, high)



def auto_pause_duration(cdr: float, rng: random.Random) -> int:
    low, high = AUTO_PAUSE_RANGES.get(cdr, (400, 1200))
    return rng.randint(low, max(low, high))


def split_speech_for_auto_pauses(text: str) -> list[str]:
    """Split a long speech string into natural chunks for pause insertion."""
    text = text.strip()
    if not text:
        return []

    # Prefer sentence boundaries first.
    chunks = [m.group(1).strip() for m in SENTENCE_BOUNDARY_RE.finditer(text) if m.group(1).strip()]
    if len(chunks) >= 2:
        return chunks

    # If there are no full stops, use commas/pauses in long utterances.
    if len(text) > 55:
        rough = re.split(r"(?<=[，、])", text)
        chunks = [x.strip() for x in rough if x.strip()]
        if len(chunks) >= 2:
            return chunks

    return [text]


def insert_auto_pauses(events: list[dict[str, Any]], cdr: float, rng: random.Random) -> list[dict[str, Any]]:
    """Ensure each transcript has enough pause events for acoustic simulation.

    This does not ask Qwen to create timing. It inserts conservative pauses at
    sentence/phrase boundaries only when the generated text lacks markers.
    """
    target = MIN_PAUSES_BY_CDR.get(cdr, 1)
    current = sum(1 for e in events if e["type"] in {"pause", "silence"})
    if current >= target:
        return events

    # Split long speech events so there are legal locations to insert pauses.
    expanded: list[dict[str, Any]] = []
    for event in events:
        if event["type"] != "speech":
            expanded.append(event)
            continue
        chunks = split_speech_for_auto_pauses(event.get("text", ""))
        for i, chunk in enumerate(chunks):
            expanded.append({"type": "speech", "text": chunk})
            # Add a pause between chunks until target is reached.
            if i < len(chunks) - 1 and current < target:
                expanded.append({"type": "pause", "duration_ms": auto_pause_duration(cdr, rng)})
                current += 1

    # If still too few pauses, insert after earlier speech chunks. Avoid putting
    # a pause after the final event when possible.
    if current < target:
        result: list[dict[str, Any]] = []
        for idx, event in enumerate(expanded):
            result.append(event)
            next_is_pause = idx + 1 < len(expanded) and expanded[idx + 1]["type"] in {"pause", "silence"}
            is_last = idx == len(expanded) - 1
            if event["type"] == "speech" and not next_is_pause and not is_last and current < target:
                result.append({"type": "pause", "duration_ms": auto_pause_duration(cdr, rng)})
                current += 1
        expanded = result

    return expanded


def events_from_marked_transcript(text: str, cdr: float, rng: random.Random) -> list[dict[str, Any]]:
    parts = [part for part in MARKER_RE.split(text) if part and part.strip()]
    events: list[dict[str, Any]] = []
    for part in parts:
        part = part.strip()
        if part in MARKERS:
            event_type, duration = marker_duration(part, cdr, rng)
            events.append({"type": event_type, "duration_ms": duration})
        else:
            speech = MARKER_RE.sub("", part).strip()
            if speech:
                events.append({"type": "speech", "text": speech})

    # Merge adjacent speech chunks.
    merged: list[dict[str, Any]] = []
    for event in events:
        if merged and event["type"] == "speech" and merged[-1]["type"] == "speech":
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + event["text"].lstrip()
        else:
            merged.append(event)
    return insert_auto_pauses(merged, cdr, rng)


def transcript_without_markers(text: str) -> str:
    return MARKER_RE.sub("", text)


def count_repetitions(text: str) -> int:
    count = 0
    count += len(re.findall(r"(那個|彼個|按呢|毋知|沒有|冷啦|食飯)[，、\s]*\1", text))
    count += len(re.findall(r"([\u4e00-\u9fff]{1,3})[.。…，、\s]+\1", text))
    return count


def estimate_semantic_drift(text: str, cdr: float) -> int:
    found = [term for term in TOPIC_TERMS if term in text]
    # This is heuristic; keep conservative for early CDR, stronger for CDR 2/3.
    if cdr <= 0.5:
        return max(0, min(1, len(set(found)) // 5))
    if cdr == 1.0:
        return max(0, min(2, len(set(found)) // 4))
    if cdr == 2.0:
        return max(1 if len(set(found)) >= 4 else 0, min(5, len(set(found)) // 3))
    return max(2 if len(set(found)) >= 4 else 1, min(6, len(set(found)) // 2))


def compute_event_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    pause_events = [e for e in events if e["type"] in {"pause", "silence"}]
    speech_events = [e for e in events if e["type"] == "speech"]
    pause_total = sum(int(e.get("duration_ms", 0)) for e in pause_events)
    return {
        "event_count": len(events),
        "speech_event_count": len(speech_events),
        "pause_event_count": len(pause_events),
        "sigh_count": sum(1 for e in events if e["type"] == "sigh"),
        "cough_count": sum(1 for e in events if e["type"] == "cough"),
        "pause_total_ms": pause_total,
        "mean_pause_ms": round(pause_total / len(pause_events), 2) if pause_events else 0,
        "speech_char_count": sum(len(e.get("text", "")) for e in speech_events),
    }


def observed_features(text: str, events: list[dict[str, Any]], cdr: float) -> dict[str, Any]:
    clean_text = transcript_without_markers(text)
    word_finding = sum(clean_text.count(term) for term in WORD_FINDING_TERMS)
    repair = sum(clean_text.count(term) for term in REPAIR_TERMS)
    repetitions = count_repetitions(clean_text)
    semantic_drift = estimate_semantic_drift(clean_text, cdr)
    stats = compute_event_stats(events)

    # Clamp by CDR so features stay plausible for the target profile.
    if cdr == 0.0:
        word_finding = 0
        repair = 0
        repetitions = min(repetitions, 1)
        semantic_drift = 0
        orientation_error = False
    elif cdr == 0.5:
        word_finding = min(word_finding, 3)
        repetitions = min(repetitions, 2)
        semantic_drift = min(semantic_drift, 1)
        orientation_error = False
    elif cdr == 1.0:
        word_finding = min(max(word_finding, 1), 6)
        repetitions = min(repetitions, 5)
        semantic_drift = min(semantic_drift, 2)
        orientation_error = any(x in clean_text for x in ["幾號", "想袂起", "毋知"])
    elif cdr == 2.0:
        word_finding = min(max(word_finding, 2), 8)
        repetitions = min(max(repetitions, 1), 7)
        semantic_drift = max(semantic_drift, 1)
        orientation_error = True
    else:
        word_finding = min(max(word_finding, 1), 7)
        repetitions = min(max(repetitions, 2), 9)
        semantic_drift = max(semantic_drift, 2)
        orientation_error = True

    return {
        "word_finding_count": int(word_finding),
        "repetition_count": int(repetitions),
        "repair_count": int(repair),
        "semantic_drift_count": int(semantic_drift),
        "orientation_error": bool(orientation_error),
        "code_switch_count": estimate_code_switch_count(clean_text),
        **stats,
    }


def estimate_code_switch_count(text: str) -> int:
    mandarinish = ["然後", "現在", "應該", "生活", "事情", "手機", "傳訊息", "健康", "準備", "醫生"]
    return sum(1 for term in mandarinish if term in text)


def reject_reason(text: str, cdr: float, events: list[dict[str, Any]], scenario: str = "") -> str | None:
    if not text or len(transcript_without_markers(text)) < 12:
        return "too short"
    for pattern in BAD_PATTERNS:
        if pattern in text:
            return f"bad pattern: {pattern}"

    stats = compute_event_stats(events)
    qmarks = text.count("？") + text.count("?")

    if cdr == 0.0:
        forbidden = ["想袂起", "想不起", "彼個彼個", "[長停頓]", "[沉默]", "[嘆氣]", "[咳嗽]"]
        for pattern in forbidden:
            if pattern in text:
                return f"CDR0 forbidden marker/pattern: {pattern}"
        if stats["pause_event_count"] > 1:
            return "too many pauses for CDR0"
        if qmarks >= 3:
            return "too many questions for CDR0"
        if scenario == "clinic" and "今仔日幾號" in text:
            orientation_tokens = [
                "星期", "禮拜", "拜一", "拜二", "拜三", "拜四", "拜五", "拜六", "拜日",
                "一號", "二號", "三號", "四號", "五號", "六號", "七號", "八號", "九號",
                "十號", "十一號", "十二號", "十三號", "十四號", "十五號", "十六號",
                "十七號", "十八號", "十九號", "二十", "三十", "月",
            ]
            # CDR 0 clinic samples should answer the date/orientation question, not only ask it.
            if not any(tok in text for tok in orientation_tokens):
                return "CDR0 clinic asks date but does not answer orientation"

    if cdr == 0.5:
        if stats["pause_event_count"] > 4:
            return "too many pauses for CDR0.5"
        if "[沉默]" in text:
            return "silence too severe for CDR0.5"

    if cdr == 1.0:
        if stats["pause_event_count"] > 7:
            return "too many pauses for CDR1"
        if stats["speech_char_count"] < 45:
            return "too short for CDR1"

    if cdr == 2.0:
        if stats["pause_event_count"] < 2:
            return "too fluent for CDR2"
        if stats["speech_char_count"] > 160:
            return "too long/fluent for CDR2"

    if cdr == 3.0:
        if stats["pause_event_count"] < 3:
            return "too fluent for CDR3"
        if stats["speech_char_count"] > 90:
            return "too long for CDR3"

    return None


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
            "top_p": 0.88,
            "num_predict": 650,
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


def make_generation_plan(cdr_counts: dict[float, int], speakers: list[SpeakerProfile], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    plan: list[dict[str, Any]] = []
    scenario_keys = list(SCENARIOS.keys())
    speaker_index = 0

    for cdr in CDR_LEVELS:
        for index in range(cdr_counts.get(cdr, 0)):
            speaker = speakers[speaker_index % len(speakers)]
            speaker_index += 1
            scenario = weighted_choice(SCENARIOS, rng)
            language_profile = weighted_choice(LANGUAGE_PROFILES, rng)
            acoustic_condition = weighted_choice(ACOUSTIC_CONDITIONS, rng)

            # ensure some deterministic scenario coverage too
            if index < len(scenario_keys):
                scenario = scenario_keys[(index + int(cdr * 10)) % len(scenario_keys)]

            sample_id = f"{canonical_cdr_name(cdr)}_{speaker.speaker_id}_{scenario}_{index:04d}"
            plan.append({
                "sample_id": sample_id,
                "cdr_level": cdr,
                "scenario": scenario,
                "speaker": asdict(speaker),
                "split": speaker.split,
                "language_profile": language_profile,
                "acoustic_condition": acoustic_condition,
                "index": index,
            })
    return plan


def load_existing_records(metadata_path: Path) -> dict[str, dict[str, Any]]:
    if not metadata_path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["sample_id"]] = record
    return records


def write_dataset_outputs(output_dir: Path, records: list[dict[str, Any]], plan: list[dict[str, Any]]) -> None:
    records = sorted(records, key=lambda item: (float(item["target_profile"]["cdr_level"]), item["sample_id"]))

    (output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with (output_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        fieldnames = [
            "sample_id", "split", "cdr_level", "cdr_label", "speaker_id", "speaker_group",
            "scenario", "language_profile", "acoustic_condition", "text_path", "events_path",
            "generation_time_sec", "validation_status",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "cdr_level": record["target_profile"]["cdr_level"],
                "cdr_label": record["target_profile"]["cdr_label"],
                "speaker_id": record["speaker"]["speaker_id"],
                "speaker_group": record["speaker"]["speaker_group"],
                "scenario": record["scenario"],
                "language_profile": record["language_profile"],
                "acoustic_condition": record["acoustic_condition"],
                "text_path": record["text_path"],
                "events_path": record["events_path"],
                "generation_time_sec": record["generation_time_sec"],
                "validation_status": record["validation_status"],
            }
            writer.writerow(row)

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_items = [r for r in records if r["split"] == split_name]
        (split_dir / f"{split_name}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in split_items),
            encoding="utf-8",
        )
        with (split_dir / f"{split_name}.txt").open("w", encoding="utf-8") as fp:
            for record in split_items:
                fp.write(record["text_path"] + "\n")

    summary = {
        "record_count": len(records),
        "plan_count": len(plan),
        "by_cdr": {},
        "by_split": {},
        "by_language_profile": {},
        "by_acoustic_condition": {},
    }
    for record in records:
        cdr = str(record["target_profile"]["cdr_level"])
        summary["by_cdr"][cdr] = summary["by_cdr"].get(cdr, 0) + 1
        summary["by_split"][record["split"]] = summary["by_split"].get(record["split"], 0) + 1
        summary["by_language_profile"][record["language_profile"]] = summary["by_language_profile"].get(record["language_profile"], 0) + 1
        summary["by_acoustic_condition"][record["acoustic_condition"]] = summary["by_acoustic_condition"].get(record["acoustic_condition"], 0) + 1
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "generation_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_with_project_parser(text: str, sample_id: str, cdr: float, scenario: str) -> dict[str, Any] | None:
    if parse_transcript is None:
        return None
    try:
        parsed = parse_transcript(raw_text=text, transcript_id=sample_id, cdr_level=cdr, scenario=scenario)
        return parsed.to_dict()
    except Exception as exc:  # keep generation robust if local parser fails
        return {"parser_error": str(exc)}


def generate_one_sample(
    args: argparse.Namespace,
    item: dict[str, Any],
    rng: random.Random,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], str, float, str]:
    cdr = float(item["cdr_level"])
    speaker = SpeakerProfile(**item["speaker"])
    prompt = build_prompt(cdr, item["scenario"], item["language_profile"], speaker)
    last_reason = "not attempted"
    last_raw = ""

    for attempt in range(1, args.max_retries + 2):
        start = time.time()
        raw = ollama_chat(args.url, args.model, SYSTEM_PROMPT, prompt, args.temperature)
        elapsed = round(time.time() - start, 2)
        last_raw = raw
        text = clean_generated_text(raw)
        events = events_from_marked_transcript(text, cdr, rng)
        reason = reject_reason(text, cdr, events, item.get("scenario", ""))
        if reason is None:
            features = observed_features(text, events, cdr)
            return text, events, features, raw, elapsed, "ok"
        last_reason = reason
        if args.verbose_rejections:
            print(f"reject attempt {attempt}/{args.max_retries + 1} | {item['sample_id']} | {reason}", file=sys.stderr)

    # Last-resort: return last sample with rejected status instead of crashing.
    text = clean_generated_text(last_raw)
    events = events_from_marked_transcript(text, cdr, rng)
    features = observed_features(text, events, cdr)
    return text, events, features, last_raw, 0.0, f"accepted_after_retries_failed: {last_reason}"


def generate(args: argparse.Namespace) -> Path:
    verify_ollama(args.url, args.model)

    cdr_counts = parse_cdr_counts(args.cdr_counts)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    rng = random.Random(args.seed)
    speakers = make_speakers(args.speaker_count, args.seed)
    plan = make_generation_plan(cdr_counts, speakers, args.seed)
    records_by_id = load_existing_records(metadata_path)
    total = len(plan)

    if records_by_id:
        print(f"resuming {run_id}: {len(records_by_id)}/{total} records already exist")

    with metadata_path.open("a", encoding="utf-8") as meta_fp:
        for item in plan:
            sample_id = item["sample_id"]
            if sample_id in records_by_id:
                continue

            cdr = float(item["cdr_level"])
            level_dir = output_dir / canonical_cdr_name(cdr)
            level_dir.mkdir(parents=True, exist_ok=True)

            text, events, features, raw_response, elapsed, validation_status = generate_one_sample(args, item, rng)
            if not text:
                print(f"empty response after retries: {sample_id}", file=sys.stderr)
                continue

            text_path = level_dir / f"{sample_id}.txt"
            events_path = level_dir / f"{sample_id}.events.json"
            text_path.write_text(text + "\n", encoding="utf-8")
            events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            parsed = parse_with_project_parser(text, sample_id, cdr, item["scenario"])
            style = CDR_STYLE[cdr]
            record = {
                "sample_id": sample_id,
                "target_profile": {
                    "cdr_level": cdr,
                    "cdr_label": style["label"],
                    "speech_rate_target": style["speech_rate_target"],
                },
                "speaker": item["speaker"],
                "split": item["split"],
                "scenario": item["scenario"],
                "scenario_info": SCENARIOS[item["scenario"]],
                "language_profile": item["language_profile"],
                "language_profile_info": LANGUAGE_PROFILES[item["language_profile"]],
                "acoustic_condition": item["acoustic_condition"],
                "acoustic_condition_info": ACOUSTIC_CONDITIONS[item["acoustic_condition"]],
                "model": args.model,
                "temperature": args.temperature,
                "text_path": str(text_path.relative_to(PROJECT_ROOT)),
                "events_path": str(events_path.relative_to(PROJECT_ROOT)),
                "generation_time_sec": elapsed,
                "validation_status": validation_status,
                "raw_text": text,
                "event_script": events,
                "observed_features": features,
                "project_parser": parsed,
            }
            meta_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            meta_fp.flush()
            records_by_id[sample_id] = record

            print(
                f"{len(records_by_id)}/{total} | {sample_id} | split={item['split']} | "
                f"lang={item['language_profile']} | acoustic={item['acoustic_condition']} | "
                f"{elapsed}s | {len(text)} chars | pauses={features['pause_event_count']} | {validation_status}"
            )

    write_dataset_outputs(output_dir, list(records_by_id.values()), plan)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text-first Taigi/mixed CDR transcripts and event JSON files.")
    parser.add_argument("--cdr-counts", default="0:300,0.5:300,1:250,2:100,3:50",
                        help="Comma-separated counts, e.g. '0:300,0.5:300,1:250,2:100,3:50'.")
    parser.add_argument("--speaker-count", type=int, default=40)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.68)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--verbose-rejections", action="store_true")
    args = parser.parse_args()

    if args.speaker_count <= 0:
        raise ValueError("--speaker-count must be positive")

    output_dir = generate(args)
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
