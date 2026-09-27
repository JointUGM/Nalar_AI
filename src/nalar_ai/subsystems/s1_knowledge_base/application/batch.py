import asyncio
from collections.abc import Awaitable, Iterable
from typing import cast


async def gather_settled[T](awaitables: Iterable[Awaitable[T]]) -> list[T]:
    """Like asyncio.gather, but every sibling finishes before the first failure is re-raised.

    A failing item must not orphan in-flight calls: their invocations belong in the ledger
    that the error envelope returns (AI-6).
    """
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return cast(list[T], results)
