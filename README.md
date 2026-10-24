| Metric | Formula | Meaning |
|---|---|---|
| `Char precision@k` | overlap chars / retrieved chars | How much of what we retrieved was the answer. Low if chunks are large and mostly irrelevant. |
| `Char recall@k` | overlap chars / ground truth chars | How much of the answer we retrieved. |
| `Char f1@k` | 2·P·R / (P+R) | Combines precision and recall. It is 0 if both are 0. |
| `Hit rate` | 1 if any top-k chunk shares at least 50% characters with the ground truth, else 0 | Did we get any of the answer. |
| `MRR` | 1 / rank of the first top-k chunk that has atleast 50% of the ground truth, else 0 | How high up the first useful chunk was. A first hit at rank 1 gives 1.0, rank 2 gives 0.5, rank 4 gives 0.25. |

Overlap chars refer to the total matching characters the chunks have with the ground truth. To avoid counting repeated characters, we take the union of our top k chunks to get total unique matching characters our retrieved text contains. For a chunk to have a valid overlap, it must share a common source with the ground truth. 

Retrieved chars refer to the total characters covered by the top-k chunks after merging.

Ground truth chars refer to the total charcaters covered by the ground truth span.

