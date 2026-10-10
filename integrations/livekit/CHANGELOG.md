# Changelog

## 0.1.0 (2026-10-10)

- First release: `duplexjev.TurnDetector`, an audio (streaming) turn detector for LiveKit Agents that asks a
  DuplexJev gateway for the turn state of the current user turn.
- Extra questions (emotion, non-verbal sound, barge-in, or your own) answered in the same request; see
  `last_answers` / `on_answers`.
- Optional backchannel opportunity from "What should the assistant do now?" (`backchannel_threshold`).
- Tested with livekit-agents 1.8.6.
