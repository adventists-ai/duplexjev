# Examples

| file | what it shows |
|---|---|
| `quickstart.py` | one clip, several option groups (`Decider.decide`) |
| `batch.py` | many clips, option groups bound to clips with `audio=` (`Decider.decide_batch`) |
| `tick_batch.py` | one tick of a batched loop: N calls x G groups in one pass, with timing |
| `server_client.py` | concurrent requests to `duplexjev serve` (`/v1/decide`, `/v1/decide_batch`) |
| `make_examples.py` | generate the project-page examples (`docs/examples.json`) from a checkpoint |

Use only audio whose license allows redistribution when publishing examples.
