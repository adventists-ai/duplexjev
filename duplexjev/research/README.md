# duplexjev/research (paper code)

| file | purpose |
|---|---|
| `contract.py` | question schema: id, text, options, validation |
| `render_letter.py` | render a question with permuted letters and check each letter is one token |
| `readout.py` | encode state + questions, one forward pass, softmax over option letters |
| `run_decision.py` | run a decision pack from the command line |
| `run_bench.py` | model loading and prompt building shared by the evaluation scripts |
| `encoder/` | Qwen3-ASR-0.6B encoder with the cross-attention fusion (variant A) and configs for A and B |
| `test_contract.py` | unit tests for the contract |

This is the code used in the paper, kept for reproducibility. The installable package is [`duplexjev`](..)
(`Decider`, `Question`). See [docs/PATHS.md](../../docs/PATHS.md).
