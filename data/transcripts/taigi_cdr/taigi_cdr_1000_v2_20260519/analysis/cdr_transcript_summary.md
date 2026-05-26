# CDR Transcript Audit Summary

This audit rates each simulated Taigi transcript against its assigned CDR folder label and flags language/structure issues for review. It is a dataset-quality audit, not a clinical diagnosis.

## Rubric

- CDR 0: coherent, oriented response with only normal hesitation.
- CDR 0.5: mild retrieval difficulty or occasional uncertainty, mostly coherent.
- CDR 1: clear memory/orientation problems, shorter answers, repeated uncertainty.
- CDR 2: moderate impairment with fragmented structure, long pauses/silence, topic drift.
- CDR 3: severe impairment with sparse, disconnected phrases and frequent silence.

## Folder Summary

| CDR | Label | Files | Heuristic Matches | Review/Fix Flags | Avg chars | Avg Taigi ratio | Avg long pauses+silence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | normal | 200 | 147 | 114 | 127.5 | 0.277 | 0.00 |
| 0.5 | questionable impairment | 200 | 79 | 143 | 167.6 | 0.272 | 0.60 |
| 1 | mild dementia | 200 | 169 | 63 | 181.2 | 0.331 | 1.23 |
| 2 | moderate dementia | 200 | 62 | 152 | 190.6 | 0.372 | 5.20 |
| 3 | severe dementia | 200 | 200 | 4 | 85.2 | 0.401 | 2.88 |

## Overall Counts

- Files audited: 1000
- CDR heuristic matches: 657
- CDR heuristic review cases: 343
- Language pass: 524
- Language review: 476
- Language fix: 0

## Common Review Flags

| Flag | Count | Meaning |
|---|---:|---|
| `heuristic_cdr_1` | 175 | Heuristic severity estimate differs from the assigned folder CDR. |
| `orthography_review` | 167 | Contains unusual or inconsistent Taigi orthography/characters that should be checked. |
| `heuristic_cdr_0.5` | 78 | Heuristic severity estimate differs from the assigned folder CDR. |
| `low_taigi_ratio` | 75 | Parsed Taigi token ratio is below 0.20. |
| `high_mandarin_ratio` | 75 | Parsed Mandarin token ratio is above 0.80. |
| `heuristic_cdr_0` | 72 | Heuristic severity estimate differs from the assigned folder CDR. |
| `heuristic_cdr_3` | 12 | Heuristic severity estimate differs from the assigned folder CDR. |
| `heuristic_cdr_2` | 6 | Heuristic severity estimate differs from the assigned folder CDR. |

## First Review Samples

| sample_id | assigned | heuristic | flags |
|---|---:|---:|---|
| `cdr_0_clinic_0003` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0008` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0018` | 0 | 0.5 | heuristic_cdr_0.5;orthography_review:几 |
| `cdr_0_clinic_0023` | 0 | 0.5 | heuristic_cdr_0.5;orthography_review:俺 |
| `cdr_0_clinic_0043` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0048` | 0 | 0 | orthography_review:嘸\|佷 |
| `cdr_0_clinic_0063` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0068` | 0 | 0.5 | heuristic_cdr_0.5;orthography_review:係 |
| `cdr_0_clinic_0078` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0083` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0093` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0108` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0128` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0143` | 0 | 0 | orthography_review:係 |
| `cdr_0_clinic_0148` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0158` | 0 | 0.5 | heuristic_cdr_0.5;orthography_review:ㄟ |
| `cdr_0_clinic_0163` | 0 | 0 | low_taigi_ratio;high_mandarin_ratio |
| `cdr_0_clinic_0173` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0183` | 0 | 0.5 | heuristic_cdr_0.5 |
| `cdr_0_clinic_0193` | 0 | 0.5 | heuristic_cdr_0.5 |

## Notes

- The folder label remains the authoritative CDR label for this simulated dataset.
- `heuristic_cdr` is a consistency check based on pauses, silences, uncertainty terms, fragmentation, and transcript length.
- `language_quality=review` usually means Taigi/Mandarin mix, unusual orthography, or a CDR mismatch worth human review.
- `language_quality=fix` is reserved for structural problems such as unbalanced markers, very short text, or metadata/folder mismatch.
