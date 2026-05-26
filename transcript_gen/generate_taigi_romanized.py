#!/usr/bin/env python3
"""
Generate pure Tâi-lô romanized Taigi dementia speech transcripts.

Strategy: Qwen generates in 台語漢字 (which it does well), then taibun
converts to Tâi-lô romanization. Biomarkers are embedded inline.

Distribution: CDR0=350, CDR0.5=300, CDR1=200, CDR2=100, CDR3=50 (1000 total)
Output: Only .txt files with inline biomarker tags.

Usage:
  python generate_taigi_romanized.py --model qwen2.5:14b
"""
from __future__ import annotations
import argparse, json, random, re, sys, time, csv
from datetime import datetime
from pathlib import Path
from typing import Any
import requests
from taibun import Converter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "transcripts" / "taigi_romanized"
CDR_LEVELS = [0, 0.5, 1, 2, 3]
DEFAULT_CDR_COUNTS = {0: 350, 0.5: 300, 1: 200, 2: 100, 3: 50}

# Biomarker tags that stay in the text (NOT converted by taibun)
BIOMARKERS = [
    "<pause>", "<long_pause>", "<silence>",
    "<word_finding>", "<repetition>", "<repair>",
    "<topic_drift>", "<filler>", "<sigh>", "<cough>",
]

# Chinese equivalents the model will produce -> biomarker tags
MARKER_MAP = {
    "[停頓]": "<pause>", "（停頓）": "<pause>", "(停頓)": "<pause>",
    "[長停頓]": "<long_pause>", "（長停頓）": "<long_pause>",
    "[沉默]": "<silence>", "（沉默）": "<silence>",
    "[找詞困難]": "<word_finding>", "[想袂起]": "<word_finding>",
    "[重複]": "<repetition>",
    "[修正]": "<repair>", "[自我修正]": "<repair>",
    "[離題]": "<topic_drift>", "[話題跳轉]": "<topic_drift>",
    "[填充]": "<filler>",
    "[嘆氣]": "<sigh>", "（嘆氣）": "<sigh>",
    "[咳嗽]": "<cough>", "（咳嗽）": "<cough>",
}

SCENARIOS = {
    "market": "講述今仔日去菜市仔買菜、和攤販講話、買菜轉去厝內。",
    "family": "講述囝仔和孫仔轉來厝內食飯，大家坐佇灶跤邊講話。",
    "morning": "講述早起後洗面、食飯、整理物件、準備出門。",
    "clinic": "在診間回答醫師問今日日期、早餐、家裡發生啥代誌。",
    "memory": "回想少年時陣的生活，例如讀冊、農忙、厝內長輩。",
    "picture": "看一張家庭或市場情境圖，描述圖內的人、動作、物件。",
}
SCENARIO_WEIGHTS = {"clinic": 250, "memory": 200, "morning": 150, "market": 150, "family": 150, "picture": 100}

BAD_PATTERNS = ["嘅", "咗", "啲", "係咪", "煮緊", "講緊", "米其林"]

CDR_STYLE = {
    0: {"label": "normal", "length": "150-210字",
        "features": "語意清楚、順序完整，只有自然口語助詞，少量自然停頓。"},
    0.5: {"label": "very_mild", "length": "130-190字",
          "features": "偶爾想袂起詞，會用那個、嗯、啊補位，但大致能回到主題。"},
    1: {"label": "mild", "length": "110-180字",
        "features": "輕度失智；明顯找詞困難、重複、停頓、輕微離題，但仍可理解。"},
    2: {"label": "moderate", "length": "70-120字",
        "features": "中度失智；句子明顯破碎，常說一半停掉；重複同一詞或同一句；話題跳走。"},
    3: {"label": "severe", "length": "25-65字",
        "features": "重度失智；只用很短、破碎、不完整的片段；頻繁中斷、沉默、重複字詞。"},
}

CDR_RULES = {
    0: """CDR 0 專用規則：
- 正常認知長輩，不能有失智症狀。
- 內容要完整、有順序、能回答情境。
- 最多1次短暫 [停頓]。不要 [長停頓] 或 [沉默]。
- 不要使用 [找詞困難]、[重複]、[離題]。
- 可自然使用 嗯、啊、齁 等語氣詞。""",
    0.5: """CDR 0.5 專用規則：
- 大致清楚完整，但偶爾找不到詞。
- 使用 1-2 次 [找詞困難]，後接「那個...」「想袂起來」。
- 可使用 0-1 次 [重複]。
- 停頓 1-3 次。不要 [沉默]。
- 可短暫離題但要自己拉回。""",
    1: """CDR 1 專用規則：
- 輕度失智，仍能理解大意。
- 使用 2-4 次 [找詞困難]。
- 使用 1-3 次 [重複]。
- 使用 1-2 次 [離題]。
- 停頓 3-6 次，可含 [長停頓]。
- 可出現 [修正]（自我修正講錯的詞）。""",
    2: """CDR 2 專用規則：
- 中度失智，句子明顯破碎。
- 使用 4-6 次 [找詞困難]。
- 使用 2-5 次 [重複]。
- 使用 2-3 次 [離題]。
- 停頓 5-9 次，含多次 [長停頓] 或 [沉默]。
- 每段話約 8-14 字就中斷。""",
    3: """CDR 3 專用規則：
- 重度失智，不要寫完整句子。
- 每段話約 2-8 字。
- 使用 3-8 次 [重複]（重複同一個詞或短語）。
- 至少 3 次 [長停頓] 或 [沉默]。
- 必須突然 [離題] 到阿母、食飯、囝仔、天氣等。
- 結尾可以未完成。""",
}

SYSTEM_PROMPT = """你是台灣本土語言專家，專精台語口語轉寫與臨床語言學。
請產生台灣65-85歲長輩自然講話的逐字稿，用台語漢字書寫。

＝＝＝ 嚴格規則 ＝＝＝
1. 100%台語，完全禁止華語。禁用：的、了、是、在、很、都、也、但是、因為、所以、然後、可以、已經、應該、如果、雖然、或者。
2. 台語對應用法：
   的→ê, 了→矣(ah), 是→是(sī), 在→佇(tī)/咧(leh), 很→真(tsin)/足(tsiok)/誠(tsiânn), 都→攏(lóng), 也→嘛(mā), 但是→毋過(m̄-koh), 因為→因為(in-uī), 所以→所以(sóo-í), 然後→了後(liáu-āu)/紲落(suà-lo̍h), 可以→會使(ē-sái), 已經→已經(í-king), 如果→若是(nā-sī)
3. 不要用羅馬字，只用漢字。
4. 不要加標題、引號、編號、說明。
5. 不要用粵語（嘅、咗、啲）。
6. 不要用簡體字。

＝＝＝ 常用台語漢字詞彙 ＝＝＝
日常：今仔日、昨昏、透早、暗時、天氣、落雨、出日頭、寒、燒
人物：阿母、阿爸、阿公、阿媽、囝仔、孫仔、新婦、翁、某、厝邊
地點：厝、灶跤、房間、菜市仔、廟口、病院、診所
動作：食飯、煮食、洗碗、洗衫、曝衫、睏、起床、行路、坐、企
食物：飯、菜、魚、肉、湯、粥、菜頭、高麗菜、豬肉、雞卵
買賣：買、賣、偌濟錢、較俗、頭家、頭家娘
形容：好、歹、大、細、濟、少、緊、慢、早、晏
連接：了後、紲落、閣、嘛、毋過、若是、所以
語氣：啊、嗯、齁、啦、咧、喔、矣、honnh
否定：毋(m̄)、袂(bē)、無(bô)、毋是(m̄-sī)、毋知(m̄-tsai)
指示：這(tsit)、彼(hit)、遮(tsia)、遐(hia)、佗位(tó-uī)

＝＝＝ 語音/認知事件標記 ＝＝＝
直接插入文中，不要另起一行：
[停頓] [長停頓] [沉默] [找詞困難] [重複] [修正] [離題] [嘆氣] [咳嗽]

只輸出逐字稿正文。"""

SPEAKER_GROUPS = [
    ("older_female", 12, "女性長輩，台語優勢，65-85歲。"),
    ("older_male", 12, "男性長輩，台語優勢，65-85歲。"),
    ("older_female_rural", 8, "女性長輩，鄉下出身，純台語。"),
    ("older_male_rural", 8, "男性長輩，鄉下出身，純台語。"),
]


def cdr_dir(cdr: float) -> str:
    return f"cdr_{str(cdr).replace('.', '_')}"


# Few-shot examples per CDR level to teach the model natural Taigi patterns
CDR_EXAMPLES = {
    0: """範例（學習風格，不要照抄）：
今仔日透早去菜市仔，彼個賣魚ê阿伯真好禮，伊講今仔日ê魚足青。我買兩尾虱目魚，閣買一斤豬肉、幾若葉菜頭。[停頓] 頭家娘算較俗，講我逐擺攏來，伊足歡喜。轉去厝了後，先洗菜，紲落來煮一鼎魚湯。孫仔放學轉來，講阿媽煮ê湯上好食。我聽著真歡喜，一家伙仔坐佇灶跤食飯，誠鬧熱。""",
    0.5: """範例（學習風格，不要照抄）：
今仔日囝仔講欲轉來食飯，我透早就去菜市仔買菜。買彼個...嗯...[找詞困難] 啊，高麗菜啦，閣有豬肉。[停頓] 轉來厝煮到一半，想袂起鹽囥佇佗位。啊，佇灶跤頂懸啦。孫仔入來講肚子枵，我講緊等咧，菜閣咧滾。[停頓] 了後逐家坐落來食飯，囝仔講我煮ê菜真好食，我嘛真歡喜。""",
    1: """範例（學習風格，不要照抄）：
今仔日...嗯...囝仔轉來，[停頓] 我去買菜，買彼個...[找詞困難] 啊，彼個叫啥？嗯...菜頭啦。[長停頓] 閣有買...[重複] 買彼個豬肉，豬肉。轉來厝欲煮食，[停頓] 毋過我袂記得鼎囥佇佗位。[找詞困難] 彼個...彼個鼎啊，佇灶跤底。[離題] 阿母以前嘛攏按呢煮食。[停頓] 啊...今仔日孫仔有來無？我想袂起來。""",
    2: """範例（學習風格，不要照抄）：
菜市仔...[長停頓] 買彼個...[找詞困難] 啊，啥物？[停頓] [重複] 買彼個，嗯...買菜。[沉默] [離題] 阿母以前攏去菜市仔。[找詞困難] 伊...伊是...[長停頓] [重複] 菜市仔。[停頓] 今仔日落雨無？[找詞困難] 啊...[離題] 囝仔轉來未？[長停頓] 食飯...[停頓] 我欲食飯。""",
    3: """範例（學習風格，不要照抄）：
菜...[長停頓] 阿母...[沉默] [重複] 阿母咧？[停頓] 食飯...[找詞困難] 彼個...[長停頓] 囝仔...[重複] 囝仔。[沉默] 寒...[離題] 欲睏...[長停頓] 袂記得...""",
}

def build_prompt(cdr: float, scenario_key: str, speaker_desc: str) -> str:
    style = CDR_STYLE[cdr]
    scenario = SCENARIOS[scenario_key]
    example = CDR_EXAMPLES[cdr]
    return f"""請產生一段台灣長輩口語逐字稿。嚴格要求100%台語，完全禁止華語。

CDR等級：{cdr}（{style['label']}）
說話者：{speaker_desc}
情境：{scenario}
認知語言特徵：{style['features']}
長度：{style['length']}

{CDR_RULES[cdr]}

{example}

嚴格要求：
- 只輸出逐字稿正文，不要任何說明。
- 100%台語漢字。禁用華語：的→用ê、了→用矣(ah)、在→用佇(tī)/咧(leh)、很→用真/足/誠、都→用攏、也→用嘛、但是→用毋過。
- 使用自然台語口語：啊、嗯、齁、啦、咧、honnh。
- 事件標記直接插在文中。
- 內容要和情境相關，講出具體ê人、物件、動作。
- 不要照抄範例，要自己創作新內容。

只輸出逐字稿正文。"""


def extract_and_replace_markers(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Extract biomarker positions, replace with placeholders for conversion."""
    markers_found = []
    # Sort by length descending to match longer patterns first
    sorted_map = sorted(MARKER_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for zh, tag in sorted_map:
        while zh in text:
            idx = text.index(zh)
            markers_found.append((idx, tag))
            text = text[:idx] + f"§{tag}§" + text[idx + len(zh):]
    return text, markers_found


def convert_to_tailo(text: str) -> str:
    """Convert 漢字 text to Tâi-lô, preserving biomarker tags."""
    converter = Converter()

    # Extract markers, replace with unique placeholders
    placeholder_map = {}
    counter = 0
    for tag in BIOMARKERS:
        pattern = f"§{tag}§"
        while pattern in text:
            ph = f"BIOMARKER{counter:04d}"
            placeholder_map[ph] = tag
            text = text.replace(pattern, ph, 1)
            counter += 1

    # Also handle raw markers the model might use directly
    for zh, tag in sorted(MARKER_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        while zh in text:
            ph = f"BIOMARKER{counter:04d}"
            placeholder_map[ph] = tag
            text = text.replace(zh, ph, 1)
            counter += 1

    # Convert remaining 漢字 to Tâi-lô
    converted = converter.get(text)

    # Restore biomarker tags
    for ph, tag in placeholder_map.items():
        # taibun may change casing/spacing around placeholders
        # Try exact match first, then case-insensitive
        if ph in converted:
            converted = converted.replace(ph, f" {tag} ")
        elif ph.lower() in converted.lower():
            idx = converted.lower().index(ph.lower())
            converted = converted[:idx] + f" {tag} " + converted[idx + len(ph):]

    # Remove any leftover CJK characters that taibun couldn't convert
    converted = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', converted)
    # Clean up spacing
    converted = re.sub(r'\s+', ' ', converted).strip()
    return converted


def clean_raw(text: str) -> str:
    """Clean model output before conversion."""
    # Remove markdown/formatting
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(('#', '**', '##', '---', '```')):
            continue
        # Skip meta lines like "CDR等級：" "說話者："
        if re.match(r'^(CDR|說話者|情境|認知|長度|共同)', s):
            continue
        lines.append(s.strip('"\'「」'))
    text = '\n'.join(lines)

    # Fix simplified -> traditional
    simp_trad = {"那个": "那個", "什么": "什麼", "怎么": "怎麼", "说": "說",
                 "来": "來", "着": "著", "买": "買", "猪": "豬", "汤": "湯"}
    for s, t in simp_trad.items():
        text = text.replace(s, t)

    # Replace leaked Mandarin with Taigi equivalents
    mandarin_to_taigi = {
        "然後": "了後", "但是": "毋過", "因為": "因為", "所以": "所以",
        "可以": "會使", "已經": "已經", "如果": "若是", "雖然": "雖然",
        "或者": "抑是", "還是": "抑是", "還有": "閣有", "還": "閣",
        "非常": "足", "特別": "誠", "什麼": "啥物", "怎麼": "按怎",
        "怎樣": "按怎", "為什麼": "為啥物", "哪裡": "佗位",
        "這個": "這個", "那個": "彼個", "這裡": "遮", "那裡": "遐",
        "他們": "𪜶", "我們": "咱", "你們": "恁",
        "爸爸": "阿爸", "媽媽": "阿母", "爺爺": "阿公", "奶奶": "阿媽",
        "孩子": "囝仔", "孫子": "孫仔", "老婆": "某", "老公": "翁",
        "廚房": "灶跤", "回家": "轉去厝", "回去": "轉去", "回來": "轉來",
        "早上": "透早", "昨天": "昨昏", "今天": "今仔日", "晚上": "暗時",
        "吃飯": "食飯", "吃": "食", "喝": "啉", "做飯": "煮食", "做菜": "煮食",
        "睡覺": "睏", "走路": "行路", "說話": "講話", "看": "看",
        "便宜": "俗", "貴": "貴", "多少錢": "偌濟錢",
        "知道": "知影", "不知道": "毋知", "記得": "記得", "忘記": "袂記得",
        "漂亮": "媠", "好看": "好看", "好吃": "好食",
        "下雨": "落雨", "天氣": "天氣", "冷": "寒", "熱": "燒",
    }
    # Sort by length (longest first) to avoid partial replacements
    for m, t in sorted(mandarin_to_taigi.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(m, t)

    # Normalize parenthetical markers
    text = text.replace("（停頓）", "[停頓]").replace("(停頓)", "[停頓]")
    text = text.replace("（長停頓）", "[長停頓]").replace("（沉默）", "[沉默]")
    text = text.replace("（嘆氣）", "[嘆氣]").replace("(嘆氣)", "[嘆氣]")
    text = text.replace("（咳嗽）", "[咳嗽]").replace("(咳嗽)", "[咳嗽]")
    text = text.replace("（找詞困難）", "[找詞困難]")
    text = text.replace("（重複）", "[重複]").replace("（修正）", "[修正]")
    text = text.replace("（離題）", "[離題]").replace("（話題跳轉）", "[離題]")

    # Remove any stray brackets not in our allowed set
    allowed_zh = set(MARKER_MAP.keys())
    def check_bracket(m):
        full = m.group(0)
        if full in allowed_zh:
            return full
        return ""
    text = re.sub(r'[\[（(][^\]）)]+[\]）)]', check_bracket, text)

    return text.strip()


def validate(cdr: float, raw_text: str, romanized: str) -> tuple[bool, str]:
    if len(raw_text) < 10:
        return False, "too short"
    for p in BAD_PATTERNS:
        if p in raw_text:
            return False, f"bad pattern: {p}"
    if len(romanized) < 20:
        return False, "romanized too short"
    # Check CDR 0 doesn't have severe markers
    if cdr == 0:
        for m in ["[長停頓]", "[沉默]", "[找詞困難]", "[重複]", "[離題]"]:
            if m in raw_text:
                return False, f"CDR 0 has severe marker: {m}"
    return True, "ok"


def count_biomarkers(text: str) -> dict[str, int]:
    counts = {}
    for tag in BIOMARKERS:
        c = text.count(tag)
        if c > 0:
            counts[tag] = c
    return counts


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


def load_existing(path: Path) -> set[str]:
    """Load sample IDs of already-generated .txt files."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done.add(rec["sample_id"])
    return done


def generate_one(args, cdr: float, scenario: str, speaker: dict) -> tuple[str, str, str]:
    """Generate one sample. Returns (romanized_text, raw_hanzi, status)."""
    prompt = build_prompt(cdr, scenario, speaker["description"])
    last_reason = "not generated"
    raw_text, romanized = "", ""

    for attempt in range(1, args.max_retries + 2):
        raw_response = ollama_chat(args.url, args.model, SYSTEM_PROMPT, prompt, args.temperature)
        raw_text = clean_raw(raw_response)
        romanized = convert_to_tailo(raw_text)

        ok, reason = validate(cdr, raw_text, romanized)
        if ok:
            return romanized, raw_text, "ok"

        last_reason = reason
        prompt = build_prompt(cdr, scenario, speaker["description"]) + \
                 f"\n\n上一版不合格原因：{reason}。請重新產生。"

    return romanized, raw_text, f"failed: {last_reason}"


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
        "cdr_counts": {str(k): v for k, v in parse_cdr_counts(args.cdr_counts).items()},
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
            romanized, raw_hanzi, status = generate_one(args, cdr, item["scenario"], item["speaker"])
            elapsed = round(time.time() - t0, 2)

            if not romanized:
                print(f"EMPTY: {sid}", file=sys.stderr)
                continue

            # Write .txt with romanized text + biomarkers
            txt_path = ldir / f"{sid}.txt"
            txt_path.write_text(romanized + "\n", encoding="utf-8")

            # Metadata record
            bm = count_biomarkers(romanized)
            record = {
                "sample_id": sid,
                "cdr_level": cdr,
                "cdr_label": CDR_STYLE[cdr]["label"],
                "scenario": item["scenario"],
                "speaker_id": item["speaker"]["speaker_id"],
                "speaker_group": item["speaker"]["group"],
                "split": item["split"],
                "text_path": str(txt_path.relative_to(PROJECT_ROOT)),
                "generation_time_sec": elapsed,
                "validation_status": status,
                "biomarkers": bm,
                "char_count": len(romanized),
            }
            mfp.write(json.dumps(record, ensure_ascii=False) + "\n")
            mfp.flush()
            existing.add(sid)

            bm_str = " ".join(f"{k}={v}" for k, v in bm.items()) if bm else "none"
            print(f"{len(existing)}/{total} | {sid} | {elapsed}s | {len(romanized)}ch | markers: {bm_str} | {status}")

    # Write manifest.csv
    all_records = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_records.append(json.loads(line))

    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "sample_id", "cdr_level", "cdr_label", "scenario", "speaker_id",
            "speaker_group", "split", "text_path", "generation_time_sec",
            "validation_status", "char_count",
        ], extrasaction="ignore")
        w.writeheader()
        w.writerows(all_records)

    # Write splits
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "val": [], "test": []}
    for r in all_records:
        by_split.setdefault(r.get("split", "train"), []).append(r)
    for sn in ["train", "val", "test"]:
        items = sorted(by_split.get(sn, []), key=lambda r: (float(r["cdr_level"]), r["sample_id"]))
        (split_dir / f"{sn}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")

    # Summary
    summary = {"total": len(all_records), "by_cdr": {}, "by_split": {}, "by_scenario": {}}
    for r in all_records:
        for f, k in [("by_cdr", str(r["cdr_level"])), ("by_split", r["split"]), ("by_scenario", r["scenario"])]:
            summary[f][k] = summary[f].get(k, 0) + 1
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone! {len(all_records)} samples saved to {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Generate Tâi-lô romanized Taigi CDR transcripts.")
    p.add_argument("--cdr-counts", default="0:350,0.5:300,1:200,2:100,3:50")
    p.add_argument("--speaker-count", type=int, default=40)
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.68)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default="")
    p.add_argument("--max-retries", type=int, default=2)
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
