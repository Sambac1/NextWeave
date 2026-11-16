import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.domain import LLMOptions
from app.llm.budget import (
    ModelBudget,
    BudgetExceeded,
    budget_scope,
    reserve_http_request,
    settle_http_response,
)


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"total_tokens": 1000},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1.5, "completion_tokens": 0},
    ],
)
def test_unusable_usage_keeps_the_reservation_on_resume(usage):
    options = LLMOptions(max_total_tokens=1000)
    ledger = ModelBudget(options)
    ticket = ledger.reserve(200, 700)
    ledger.settle(ticket, usage, 200)
    resumed = ModelBudget(options, json.loads(json.dumps(ledger.report())))
    assert resumed.report()["accounted_tokens_upper"] == 900
    with pytest.raises(BudgetExceeded):
        resumed.reserve(100, 100)


def test_parallel_http_reservations_are_durable_before_dispatch_and_credential_free():
    writes = []
    sent = []
    ledger = ModelBudget(
        LLMOptions(max_total_tokens=1000),
        on_change=lambda v: writes.append(json.loads(json.dumps(v))),
    )

    async def transport(request):
        assert writes[-1]["requests"][-1]["status"] == "reserved_or_unknown"
        sent.append(1)
        await asyncio.sleep(0.02)
        return httpx.Response(
            200, json={"usage": {"prompt_tokens": 60, "completion_tokens": 10}}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport),
            event_hooks={
                "request": [reserve_http_request],
                "response": [settle_http_response],
            },
        ) as client:
            with budget_scope(ledger):
                return await asyncio.gather(
                    *(
                        client.post(
                            "https://provider.example/chat/completions",
                            json={
                                "messages": [{"role": "user", "content": "small"}],
                                "max_tokens": 700,
                            },
                            headers={"Authorization": "Bearer PRIVATE_KEY"},
                        )
                        for _ in range(2)
                    ),
                    return_exceptions=True
                )

    results = asyncio.run(run())
    assert len(sent) == 2 and all(isinstance(v, httpx.Response) for v in results)
    assert ledger.report()["accounted_tokens_upper"] == 140
    assert ledger.reservation_waits >= 1
    assert "PRIVATE_KEY" not in json.dumps(writes)


def test_price_configuration_is_explicit_and_cannot_be_changed_to_bypass_resume_cap():
    with pytest.raises(ValidationError):
        LLMOptions(max_cost_usd=0.1)
    options = LLMOptions(
        max_cost_usd=0.0001, input_price_per_million=1, output_price_per_million=1
    )
    ledger = ModelBudget(options)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(100, 100)
    assert ledger.calls == 0
    ledger.reserve(10, 10)
    with pytest.raises(ValueError, match="original unit prices"):
        ModelBudget(
            LLMOptions(input_price_per_million=0, output_price_per_million=0),
            ledger.report(),
        )
