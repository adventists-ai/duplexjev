# Option order: do answers and confidence change when the options are shuffled?

DuplexJev reads each answer from the next-token distribution over the option letters, so the order in which the options
are listed could in principle move the answer. This folder measures that on the 2,062 same-item questions of
[`paraling_compare`](../paraling_compare) that have 2–10 options (the 39 items with 16 options and the one
single-option item are skipped).

**Setup.** DuplexJev-32B-Para v1.3 served with vLLM, the same prompt format as `duplexjev gateway` / `VLLMClient`, one option
order per request (no averaging). Items with up to 4 options are asked in **every** order (24 for 4 options); items with
5–10 options in 12 seeded random orders including the original one. 33,508 requests in total.

**Results** (`python analyse.py`):

| set | items | orders | answer differs from the item's majority answer | items with any change | accuracy, one order | accuracy, orders averaged | median spread of the majority answer's probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMO (emotion, 4-way) | 200 | 4,800 | 0.65% | 3.0% | 90.0 | 90.0 | 0.002 |
| POSN-zh / POSN-en (stressed word) | 200 / 174 | 3,360 / 4,488 | 0.60% / 0.60% | 2.5% / 3.4% | 96.3 / 90.9 | 97.0 / 90.8 | 0.002 / 0.008 |
| DEBATE (meaning of stress) | 200 | 2,800 | 1.18% | 5.0% | 94.6 | 94.5 | 0.004 |
| StressTest | 200 | 400 | 5.75% | 11.5% | 84.2 | 84.0 | 0.038 |
| MSPB | 128 | 3,620 | 6.27% | 22.7% | 75.6 | 77.3 | 0.149 |
| speaking style (EARS) | 200 | 4,800 | 4.21% | 21.0% | 78.9 | 79.0 | 0.115 |
| LISTEN neutral text / emotional text / mismatched / no words | 200 / 161 / 199 / 200 | 2,400 / 1,932 / 2,508 / 2,400 | 2.75 / 6.31 / 6.58 / 8.12% | 10.5 / 23.6 / 27.1 / 29.5% | 43.7 / 38.4 / 43.9 / 27.6 | 43.5 / 37.9 / 44.2 / 27.0 | 0.079–0.193 |
| **all** | **2,062** | **33,508** | **3.32%** | **14.2%** | **69.6** | **69.6** | **0.048** |

- **Confident answers do not move.** On items where the majority answer's mean probability is at least 0.8, the answer
  changes in 0.01% of orders. Changes happen on items the model is unsure about (mostly the hard LISTEN conditions and MSPB),
  which is also where its probability is spread over several options.
- **No position bias.** On 4-option items the chosen letter is A / B / C / D in 24.9 / 25.1 / 25.2 / 24.8% of answers,
  and accuracy is 86.3 / 86.6 / 86.9 / 86.2% when the correct option is shown at A / B / C / D.
- **Accuracy does not depend on the order** (69.6 for a single order and for the average over orders). Averaging over
  orders mainly makes the confidence of uncertain answers more stable. The PyTorch `Decider` can do this today (`n_perm`);
  the vLLM client and gateway use one deterministic order per question.

| file | what |
|---|---|
| `run.py` | `python run.py ../paraling_compare/items/items.jsonl <audio dir> out.jsonl http://localhost:8000/v1` (`CONC=8` parallel requests; resumable) |
| `analyse.py` | the table above from `results_v1.3.jsonl.gz` or a new run |
| `results_v1.3.jsonl.gz` | one line per (item, order): the order shown (`perm[i]` = option shown at letter i), the probability of every option in the item's original order, gold |

The audio is not redistributed; see [`paraling_compare`](../paraling_compare) for where each clip comes from.
