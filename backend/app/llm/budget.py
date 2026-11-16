"""Context-local accounting at the actual HTTP boundary, including retries."""

from __future__ import annotations

import contextvars
import asyncio
import json
import math
from contextlib import contextmanager


class BudgetExceeded(RuntimeError):
    pass


CURRENT = contextvars.ContextVar("llm_request_budget", default=None)
HTTP_TICKET = contextvars.ContextVar("llm_http_ticket", default=None)


class ModelBudget:
    def __init__(self, options, previous=None, on_change=None):
        self.max_calls = options.max_model_calls
        self.max_tokens = options.max_total_tokens
        self.max_cost = options.max_cost_usd
        self.input_price = options.input_price_per_million
        self.output_price = options.output_price_per_million
        self.cached_price = options.cached_input_price_per_million
        previous = previous or {}
        self.on_change = on_change
        self.pricing = {
            "input": self.input_price,
            "output": self.output_price,
            "cached": self.cached_price,
        }
        if previous.get("requests") and previous.get("pricing") != self.pricing:
            raise ValueError("Resumed experiments must keep their original unit prices. You may raise the budget limit.")
        self.requests = list(previous.get("requests", []))
        self.prompt_tokens = int(previous.get("prompt_tokens", 0))
        self.completion_tokens = int(previous.get("completion_tokens", 0))
        self.inflight = set()
        self.changed = asyncio.Event()
        self.reservation_waits = 0

    async def reserve_when_available(self, input_upper, output_upper):
        """Wait only for active HTTP calls; unknown historical usage stays charged."""
        while True:
            changed = self.changed
            try:
                ticket = self.reserve(input_upper, output_upper)
            except BudgetExceeded:
                if not self.inflight or (self.max_calls is not None and self.calls >= self.max_calls):
                    raise
                self.reservation_waits += 1
                await changed.wait()
            else:
                self.inflight.add(ticket["number"])
                return ticket

    def finish_http(self, ticket):
        self.inflight.discard(ticket["number"])
        self.changed.set()
        self.changed = asyncio.Event()

    @property
    def calls(self):
        return len(self.requests)

    def reserve(self, input_upper, output_upper):
        self.check()
        tokens = input_upper + output_upper
        price = (
            input_upper * max(self.input_price or 0, self.cached_price or 0) / 1e6
            + output_upper * (self.output_price or 0) / 1e6
        )
        totals = self.report()
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded("The model request limit was reached. Completed progress was saved.")
        if self.max_tokens is not None and totals["accounted_tokens_upper"] + tokens > self.max_tokens:
            raise BudgetExceeded(
                "The token budget cannot cover the next request. Reduce the input or raise the budget."
            )
        if (
            self.max_cost is not None
            and totals["accounted_cost_upper_usd"] + price > self.max_cost + 1e-12
        ):
            raise BudgetExceeded("The cost budget cannot cover the next request. Completed progress was saved.")
        ticket = {
            "number": self.calls + 1,
            "status": "reserved_or_unknown",
            "reserved_tokens": tokens,
            "reserved_cost_usd": price,
        }
        self.requests.append(ticket)
        self.flush()
        return ticket

    def flush(self):
        if self.on_change:
            self.on_change(self.report())

    def check(self):
        totals = self.report()
        if (self.max_tokens is not None and totals["accounted_tokens_upper"] > self.max_tokens) or (
            self.max_cost is not None
            and totals["accounted_cost_upper_usd"] > self.max_cost + 1e-12
        ):
            raise BudgetExceeded("Recorded usage exceeds the current budget. Raise the total budget to resume.")

    def settle(self, ticket, usage, status):
        ticket["http_status"] = status
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            # Never treat cancellation, protocol errors or missing usage as free.
            self.finish_http(ticket)
            self.flush()
            return
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        hit = usage.get(
            "prompt_cache_hit_tokens",
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0,
        )
        if type(hit) is not int or hit < 0:
            hit = 0
        hit = min(max(hit, 0), prompt)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        cost = (
            (prompt - hit) * (self.input_price or 0)
            + hit
            * (
                self.cached_price
                if self.cached_price is not None
                else self.input_price or 0
            )
            + completion * (self.output_price or 0)
        ) / 1e6
        ticket.update(
            status="reported",
            reported_tokens=prompt + completion,
            reported_cost_usd=cost,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        self.finish_http(ticket)
        self.flush()

    def report(self):
        unknown = [r for r in self.requests if r["status"] != "reported"]
        return {
            "actual_http_requests": self.calls,
            "reservation_waits": self.reservation_waits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "unknown_usage_requests": len(unknown),
            "accounted_tokens_upper": sum(
                r.get("reported_tokens", r["reserved_tokens"]) for r in self.requests
            ),
            "accounted_cost_upper_usd": sum(
                r.get("reported_cost_usd", r["reserved_cost_usd"])
                for r in self.requests
            ),
            "currency_estimate_available": self.input_price is not None
            and self.output_price is not None,
            "limits": {
                "max_calls": self.max_calls,
                "max_total_tokens": self.max_tokens,
                "max_cost_usd": self.max_cost,
            },
            "requests": self.requests,
            "pricing": self.pricing,
        }


@contextmanager
def budget_scope(budget):
    existing = CURRENT.get()
    token = CURRENT.set(existing or budget)
    try:
        yield existing or budget
    finally:
        CURRENT.reset(token)


async def reserve_http_request(request):
    budget = CURRENT.get()
    if budget is None or not request.url.path.endswith("/chat/completions"):
        return
    payload = json.loads(request.content)
    # UTF-8 bytes are a deliberately conservative token bound; no tokenizer or
    # provider-side cache discount is assumed before the response arrives.
    input_upper = len(
        json.dumps(payload.get("messages", []), ensure_ascii=False).encode()
    )
    ticket = await budget.reserve_when_available(
        input_upper, int(payload.get("max_tokens", 8192))
    )
    request.extensions["eip_budget"] = (budget, ticket)
    HTTP_TICKET.set((budget, ticket))


async def settle_http_response(response):
    owned = response.request.extensions.get("eip_budget")
    if not owned:
        return
    budget, ticket = owned
    await response.aread()
    try:
        usage = response.json().get("usage") or {}
    except (ValueError, AttributeError):
        usage = {}
    budget.settle(ticket, usage, response.status_code)
