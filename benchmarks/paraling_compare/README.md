# Paralinguistics: DuplexJev vs. open and closed speech models (same items)

The numbers in the README's "Against closed models, realtime flagships included" table come from the files in this folder.

| path | what |
|---|---|
| `items/items.jsonl` | the 2,102 questions: set, item id, language, audio file name, question, options (in the order every closed model saw them), gold option |
| `items/listen_sources.json` | LISTEN item id → source corpus (used for the "unseen corpora only" row) |
| `answers/duplexjev_qwen3omni.json` | set → item id → `[Qwen3-Omni-30B-A3B prediction, DuplexJev-32B-Para v1.2 prediction]` (option index). Both are single-token readouts with options rotated and averaged |
| `answers/<model>.jsonl` | one line per item for each closed model: parsed prediction, gold, the model's raw answer (first 200 characters) |
| `score.py` | prints the comparison table on the items every listed model answered |
| `run_gemini.py` | Gemini via `generate_content` (audio + question in one request) |
| `run_openai.py` | OpenAI chat completions with audio input (`gpt-audio-1.5`) |
| `run_realtime.py` | OpenAI Realtime over websocket (`gpt-realtime-2.1`): one session per item, text output, VAD off, audio and question in one user message |
| `run_glive.py` | Gemini Live (`gemini-3.8-live`): the question first, then the audio streamed as realtime input inside one manual activity (VAD off). The model can only answer by voice, so its output transcription is parsed |

```bash
python score.py                                   # all models
python score.py gemini-3.8-live gpt-realtime-2.1  # only items these two answered
```

**Protocol.** Each closed model is asked once per item, with the options in their original order, and the first option
letter in its reply is taken as its answer; a reply with no option letter counts as wrong (GPT-audio-1.5 about 2%,
Gemini 3.8 Live 0.5%, the others about 0). `answers/gemini-3.8-live-inline.jsonl` is an earlier run that sent the audio
inline with the question in one client turn; Gemini Live often replied that it had no audio (7.5% of items), so the
table uses the streamed run.

**Audio.** The clips are not redistributed. They come from public corpora (LISTEN on Hugging Face `VibeCheck1/LISTEN_full`,
ESD, CREMA-D, StressTest, EARS, MSP-Podcast-based MSPB, and our stress sets) and are identified by the item ids; clips
were converted to 16 kHz mono Ogg Opus and LISTEN clips longer than 30 s were cut to their first 30 s. Put them in
`audio/` (or point `AUDIO_DIR` at them) to re-run a model. Keys are read from `GEMINI_API_KEY` / `OPENAI_API_KEY`.

**Caveats.** Sets marked † in the README come from corpora in DuplexJev's training data (held-out speakers or sentences).
LISTEN's neutral-text set includes ESD and CREMA-D, which are in that data too; the "unseen corpora only" row
(TESS, SAVEE, RAVDESS) excludes them. Both GPT models and Gemini 3.8 Live mostly answer "neutral" to emotion questions.
