# Changelog

## 0.1.1 (2026-10-12)

- The turn probability is now P(finished) + P(backchannel): a short "ok" / "yeah" with nobody else talking also
  ends the user's turn ("still talking" and "asking to wait" still hold the turn). On LiveKit eot-bench
  (DuplexJev-32B-Turn, English) false cut-offs at a 300 ms budget drop from 37% to 30%; Smart Turn test-set accuracy
  is unchanged.

## 0.1.0 (2026-10-10)

- First release: `duplexjev.TurnDetector`, an audio (streaming) turn detector for LiveKit Agents that asks a
  DuplexJev gateway for the turn state of the current user turn.
- Extra questions (emotion, non-verbal sound, barge-in, or your own) answered in the same request; see
  `last_answers` / `on_answers`.
- Optional backchannel opportunity from "What should the assistant do now?" (`backchannel_threshold`).
- Tested with livekit-agents 1.8.6.
