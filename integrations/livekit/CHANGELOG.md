# Changelog

## 0.3.2 (2026-10-13)

- Sends an `X-DuplexJev-Client: livekit-plugins-duplexjev/<version>` header (no personal data), so trial-API usage can be counted per integration.

## 0.3.1 (2026-10-13)

- Logs a one-time hint when the free trial API is used: it runs DuplexJev-4B-Para, so
  its turn-taking does not match the published eot-bench numbers (DuplexJev-32B-Para v1.2 behind your own
  `duplexjev gateway` >= 0.6).
- README: how to reproduce the eot-bench numbers; the self-hosting example now serves DuplexJev-32B-Para v1.2 with
  `--rate 1000000`; eot-bench figure for the default setting updated to the full run (20.0%).

## 0.3.0 (2026-10-13)

- `readout="act"`: ask "What should the assistant do now?" and report P(reply now) + P(short acknowledgement) as the
  end-of-turn probability (also gives the backchannel probability from the same answer). On LiveKit eot-bench
  (English, 32B-Para v1.2, with the agent's previous message) false cut-offs at 300 ms: 18.4% vs 19.7%.
- `last_turn_class` (finished / not_finished / backchannel / hesitating, or reply / keep_listening / acknowledge).

## 0.2.0 (2026-10-13)

- Turn-taking context: `detector.attach(session)` (or `detector.set_assistant_said(text)`) sends the agent's previous
  message with every request. Needs a gateway from `duplexjev>=0.6`. On LiveKit eot-bench (English,
  DuplexJev-32B-Para v1.2) false cut-offs at a 300 ms budget drop from 23.5% to 19.7%.

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
