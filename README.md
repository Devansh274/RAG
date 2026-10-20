| Metric | Formula | Meaning |
|---|---|---|
| `char_precision@k` | overlap_chars / retrieved_chars | How much of what you retrieved was the answer. Low if chunks are large and mostly irrelevant. |
| `char_recall@k` | overlap_chars / gt_chars | How much of the answer you retrieved. |
| `char_f1@k` | 2·P·R / (P+R) | Combines precision and recall. It is 0 if both are 0. |
| `hit_rate` | 1 if any top-k chunk shares at least one character with the ground truth, else 0 | Did you get any of the answer. |
| `mrr` | 1 / rank of the first top-k chunk that overlaps the ground truth, else 0 | How high up the first useful chunk was. A first hit at rank 1 gives 1.0, rank 2 gives 0.5, rank 4 gives 0.25. |

