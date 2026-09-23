# Examples

| file | what it shows |
|---|---|
| `quickstart_text.py` | typed decisions over transcripts with any open LLM |
| `quickstart_speech.py` | decisions straight from audio with a speech checkpoint (Ultravox format, DuplexJev adapters) |
| `tick_batch.py` | one tick of a batched loop: N calls x Q questions in one pass, with timing |
| `server_client.py` | many concurrent requests to `duplexjev serve`, answered per tick |
| `make_examples.py` | generate the project-page examples (`docs/examples.json`) from a checkpoint |

Use only audio whose license allows redistribution when publishing examples.
