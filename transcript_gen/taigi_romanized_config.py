"""Configuration for pure Taigi Tâi-lô romanized transcript generation."""

CDR_LEVELS = [0, 0.5, 1, 2, 3]

DEFAULT_CDR_COUNTS = {0: 350, 0.5: 300, 1: 200, 2: 100, 3: 50}

SCENARIOS = {
    "market": "Kóng kin-á-ji̍t khì tshài-tshī-á bé-tshài, kah thâu-ke kóng-uē, bé-tshài tńg-khì tshù-lāi.",
    "family": "Kóng kiánn-jî kah sun-á tńg-lâi tshù-lāi tsia̍h-pn̄g, ta̍k-ke tsē tī tsàu-kha-piⁿ kóng-uē.",
    "morning": "Kóng tsá-khí-lâi sé-bīn, tsia̍h-pn̄g, tsíng-lí mi̍h-kiānn, tsún-pī tshut-mn̂g.",
    "clinic": "Tī tsín-kan hê-tap i-sing mn̄g kin-á-ji̍t ji̍t-kî, tsá-khí tsia̍h siánn, tshù-lāi huat-sing siánn tāi-tsì.",
    "memory": "Hê-sióng siàu-liân sî-tsūn ê sing-ua̍h, pí-lūn tha̍k-tsheh, lông-bâng, tshù-lāi tiúnn-puè.",
    "picture": "Khuànn tsi̍t-tiunn ka-tîng á-sī tshài-tshī-á ê tôo, biâu-su̍t tôo-lāi ê lâng, tōng-tsok, mi̍h-kiānn.",
}

SCENARIO_WEIGHTS = {"clinic": 250, "memory": 200, "morning": 150, "market": 150, "family": 150, "picture": 100}

CDR_STYLE = {
    0: {
        "label": "normal",
        "length": "8-12 kù (sentences)",
        "features": "Gí-ì tshing-tshó, sūn-sū uân-tsíng, tsí-ū tsū-jiân kháu-gí助詞, tsió-liōng thîng-tùn.",
        "pause_markers": "0-1 [thîng]",
    },
    0.5: {
        "label": "very_mild",
        "length": "7-10 kù",
        "features": "Ngóo-ní tshē-bô-jī, ē iōng 'hit-ê...', 'enn...', 'ah...' pó͘-uī, tāi-khài ē tńg-lâi tsú-tê.",
        "pause_markers": "1-3 [thîng] or [tn̂g-thîng]",
    },
    1: {
        "label": "mild",
        "length": "6-9 kù",
        "features": "Bîng-hián tshē-jī khùn-lân, tîng-ho̍k, thîng-tùn, khin-bî lī-tê, m̄-koh iáu ē-tàng lí-kái.",
        "pause_markers": "3-6 [thîng]/[tn̂g-thîng]",
    },
    2: {
        "label": "moderate",
        "length": "4-7 kù (short fragments)",
        "features": "Kù-tsí phò-suì, kóng tsi̍t-puànn tiām--khì, tîng-ho̍k kâng-khoán jī-sû, uē-tê thiàu-tsáu.",
        "pause_markers": "5-9 [thîng]/[tn̂g-thîng]/[tiām-tsīng]",
    },
    3: {
        "label": "severe",
        "length": "2-5 very short fragments",
        "features": "Bô uân-tsíng kù, kan-na phìnn-tuānn jī-sû, pîn-huân tiong-tuān, tîng-ho̍k, tu̍t-jiân uānn uē-tê.",
        "pause_markers": "5-12 [thîng]/[tn̂g-thîng]/[tiām-tsīng]",
    },
}

CDR_PROFILE = {
    0: {"pause_count": (0, 1), "pause_range_ms": (200, 700), "word_finding": (0, 0), "repetition": (0, 0), "drift": (0, 0), "speech_rate": 1.0},
    0.5: {"pause_count": (1, 3), "pause_range_ms": (500, 1200), "word_finding": (1, 2), "repetition": (0, 1), "drift": (0, 1), "speech_rate": 0.92},
    1: {"pause_count": (3, 6), "pause_range_ms": (800, 2500), "word_finding": (2, 5), "repetition": (1, 3), "drift": (1, 2), "speech_rate": 0.85},
    2: {"pause_count": (5, 9), "pause_range_ms": (1500, 4500), "word_finding": (4, 8), "repetition": (2, 5), "drift": (2, 4), "speech_rate": 0.75},
    3: {"pause_count": (5, 12), "pause_range_ms": (2500, 7000), "word_finding": (3, 8), "repetition": (3, 8), "drift": (3, 6), "speech_rate": 0.65},
}

CDR_RULES = {
    0: """CDR 0 rules:
- Normal cognition elder. NO dementia symptoms.
- Content complete, ordered, coherent.
- At most 1 brief [thîng]. No [tn̂g-thîng] or [tiām-tsīng].
- No heavy repetition or topic drift.""",
    0.5: """CDR 0.5 rules:
- Mostly clear, occasional word-finding difficulty.
- 1-2 instances of 'hit-ê...', 'siūnn-bē-khí-lâi'.
- May briefly drift but returns to topic.
- Few pauses, not severely fragmented.""",
    1: """CDR 1 rules:
- Mild dementia. Still understandable overall.
- Clear word-finding difficulty, repetition, pausing, slight topic drift.
- Use 'hit-ê... hit-ê kiò siánn', 'siūnn-bē-khí-lâi'.
- Sentences shorter but not just fragments like CDR 3.""",
    2: """CDR 2 rules:
- Moderate dementia. Sentences clearly fragmented.
- Each speech segment ~4-8 words. 3+ interruptions or repetitions.
- At least 2 associative topic jumps (e.g. scenario -> food, family, childhood, health).
- Use many [tn̂g-thîng] or [tiām-tsīng].""",
    3: """CDR 3 rules:
- Severe dementia. NO complete sentences or narratives.
- Each fragment ~2-5 words only.
- At least 3 [tn̂g-thîng] or [tiām-tsīng].
- Sudden topic switches to a-bú, tsia̍h-pn̄g, kiánn-jî, thinn-khì.
- May end incomplete.
- Example style (do NOT copy): 'Tshài-tshī-á... a-bú leh? Enn... tsia̍h-pn̄g... kiánn-jî, hit-ê... kuânn--lah... bē-kì-tit...'""",
}

SYSTEM_PROMPT = """Lí sī Tâi-uân pún-thóo gí-giân, Tâi-gí kháu-gí tsuán-siá, lîm-tshn̂g gí-giân-ha̍k tsuan-ka.
Tshiánn sán-sing Tâi-uân tiúnn-puè tsū-jiân kóng-uē ê tsia̍t-jī-kó, iōng Tâi-lô lô-má-jī (Taiwanese Romanization) siá.

Important rules:
- Output MUST be 100% in Tâi-lô romanization. NO Chinese characters (漢字) at all.
- Use standard Tâi-lô: proper tone marks (á, à, â, ā, a̍, a̋), nn for nasals, tsh for aspirated affricates.
- Common words: kin-á-ji̍t, tshài-tshī-á, beh, bē, m̄-tsai, siánn-mih, án-ne, tī, i, leh, lóng, koh, tsiok, tsiânn, a-bú, a-pah, kiánn-jî, sun-á, tshù, tsàu-kha, tsia̍h-pn̄g, tńg-khì.
- Fillers: enn, ah, hit-ê, tō-sī, honn, lah, leh, hioh.
- Allowed markers: [thîng], [tn̂g-thîng], [tiām-tsīng], [thàn-khùi], [ka-sàu].
- Sound like a 65-85 year old Taiwanese elder chatting naturally. NOT written prose.
- Do NOT add titles, labels, numbering, quotes, or explanations.
- Output ONLY the transcript text in Tâi-lô romanization."""

SPEAKER_GROUPS = [
    ("older_female_taigi", 12, "Lú-sìng tiúnn-puè, Tâi-gí uī-tsú, 65-85 huè."),
    ("older_male_taigi", 12, "Lâm-sìng tiúnn-puè, Tâi-gí uī-tsú, 65-85 huè."),
    ("older_female_rural", 8, "Lú-sìng tiúnn-puè, tsng-kha tshut-sin, Tâi-gí ûi-it gí-giân."),
    ("older_male_rural", 8, "Lâm-sìng tiúnn-puè, tsng-kha tshut-sin, Tâi-gí ûi-it gí-giân."),
]

ALLOWED_MARKERS = ["[thîng]", "[tn̂g-thîng]", "[tiām-tsīng]", "[thàn-khùi]", "[ka-sàu]"]

BAD_PATTERNS = ["嘅", "咗", "啲", "係", "的", "了", "是", "我", "你", "他", "她", "們", "這", "那", "什", "麼", "嗎", "吧", "呢"]
