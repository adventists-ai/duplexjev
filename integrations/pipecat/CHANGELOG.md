# Changelog

## 0.3.0 (2026-10-13)

- Turn-taking context: `analyzer.set_assistant_said(text)` (or put `DuplexJevContextTracker(analyzer)` after your LLM)
  sends the assistant's previous message with every request. Needs a gateway from `duplexjev>=0.6`. On LiveKit
  eot-bench (English, DuplexJev-32B-Para v1.2) false cut-offs at a 300 ms budget drop from 23.5% to 19.7%.

## 0.2.0 (2026-10-12)

- The turn probability is now P(finished) + P(backchannel): a short "ok" / "yeah" with nobody else talking also
  ends the user's turn ("still talking" and "asking to wait" still hold the turn). On LiveKit eot-bench
  (DuplexJev-32B-Turn, English) false cut-offs at a 300 ms budget drop from 37% to 30%; Smart Turn test-set accuracy
  is unchanged.

## 0.1.0 (2026-10-10)

- First release: `DuplexJevTurnAnalyzer`, a Pipecat `BaseSmartTurn` that asks a DuplexJev gateway for the turn state.
- Extra questions (emotion, non-verbal sound, barge-in, or your own) answered in the same request; see `last_answers`.
- Tested with pipecat-ai 1.12.0.
