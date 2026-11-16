"""Request-loop-scoped HTTPX clients shared by concurrent model calls."""

from __future__ import annotations

import asyncio
import weakref

import httpx
from app.llm.budget import reserve_http_request, settle_http_response, HTTP_TICKET


_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class _ClientLease:
    """Async context facade that reuses a client without closing the shared pool."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self

    async def post(self, *args, **kwargs):
        token = HTTP_TICKET.set(None)
        try:
            return await self.client.post(*args, **kwargs)
        finally:
            owned = HTTP_TICKET.get()
            if owned:
                budget, ticket = owned
                budget.finish_http(ticket)
            HTTP_TICKET.reset(token)

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


def shared_http_client(*, timeout: float, trust_env: bool = True) -> _ClientLease:
    loop = asyncio.get_running_loop()
    clients = _clients.setdefault(loop, {})
    key = (float(timeout), bool(trust_env))
    client = clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout, trust_env=trust_env,
            event_hooks={"request":[reserve_http_request],"response":[settle_http_response]})
        clients[key] = client
    return _ClientLease(client)


async def close_shared_http_clients() -> None:
    clients = [client for group in list(_clients.values()) for client in group.values()]
    _clients.clear()
    if clients:
        await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
