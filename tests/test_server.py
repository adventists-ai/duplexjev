"""All requests that arrive within one tick are answered by one batched call."""
import asyncio

import pytest

pytest.importorskip("fastapi")
from duplexjev.server import TickBatcher  # noqa: E402


class FakeDecider:
    def __init__(self):
        self.calls = []
        self.last_stats = {}

    def _decide_items(self, items, **kw):
        self.calls.append(len(items))
        self.last_stats = {"clips": len(items)}
        return [{"echo": {"answer": it["audio"], "confidence": 1.0, "probs": {it["audio"]: 1.0}}} for it in items]


def test_one_call_per_tick():
    async def run():
        d = FakeDecider()
        b = TickBatcher(d, tick_ms=50)
        b.start()
        one = [b.submit([{"audio": f"call-{i}"}]) for i in range(30)]
        many = [b.submit([{"audio": f"multi-{i}-{j}"} for j in range(5)]) for i in range(2)]
        res = await asyncio.gather(*one, *many)
        return d, res

    d, res = asyncio.run(run())
    assert d.calls == [40]  # 30 single-clip requests + 2 requests of 5 clips, one pass
    assert [r["answers"][0]["echo"]["answer"] for r in res[:30]] == [f"call-{i}" for i in range(30)]
    assert [a["echo"]["answer"] for a in res[31]["answers"]] == [f"multi-1-{j}" for j in range(5)]
    assert {r["tick"] for r in res} == {1} and all(r["clips_in_pass"] == 40 for r in res)
