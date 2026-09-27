import asyncio
from collections.abc import Awaitable, Callable


async def retry_transient[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_transient: Callable[[BaseException], bool],
    max_attempts: int,
    base_delay_s: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `operation`, retrying transient failures with exponential backoff (0.5s, 1s, 2s…)."""
    attempt = 1
    while True:
        try:
            return await operation()
        except Exception as exc:
            if not is_transient(exc) or attempt >= max_attempts:
                raise
            await sleep(base_delay_s * 2 ** (attempt - 1))
            attempt += 1
