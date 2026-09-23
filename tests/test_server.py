"""All requests that arrive within one tick are answered by one batched call."""
import asyncio

import pytest

pytest.importorskip("fastapi")
from duplexjev.server import TickBatcher  # noqa: E402


class FakeDecider:
    is_audio = False

    def __init__(self):
        self.calls = []
        self.last_stats = {}

    def decide(self, items, mode="packed", max_items=None):
        self.calls.append(len(items))
        self.last_stats = {"items": len(items)}
        return [{"echo": {"answer": it["text"], "confidence": 1.0, "probs": {it["text"]: 1.0}}} for it in items]


def test_one_call_per_tick():
    async def run():
        d = FakeDecider()
        b = TickBatcher(d, tick_ms=50)
        b.start()
        res = await asyncio.gather(*[b.submit({"text": f"call-{i}"}) for i in range(40)])
        return d, res

    d, res = asyncio.run(run())
    assert d.calls == [40]
    assert [r["answers"]["echo"]["answer"] for r in res] == [f"call-{i}" for i in range(40)]
    assert {r["tick"] for r in res} == {1} and all(r["batch_items"] == 40 for r in res)
