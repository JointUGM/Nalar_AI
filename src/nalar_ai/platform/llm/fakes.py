from collections import deque
from collections.abc import Callable, Iterable

from nalar_ai.platform.llm.ports import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    PermanentLLMError,
)

Reply = str | BaseException | LLMResponse


class ScriptedLLM:
    """LLMPort for tests, offline evals and the `fake` provider.

    Replies come from the queue in order, or from `route(request)` when given, which is
    deterministic under asyncio.gather. Every request is recorded in `.requests`.
    """

    def __init__(
        self, replies: Iterable[Reply] = (), route: Callable[[LLMRequest], Reply] | None = None
    ) -> None:
        self._replies: deque[Reply] = deque(replies)
        self._route = route
        self.requests: list[LLMRequest] = []

    def queue(self, *replies: Reply) -> None:
        self._replies.extend(replies)

    @property
    def remaining(self) -> int:
        return len(self._replies)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._route is not None:
            reply = self._route(request)
        elif self._replies:
            reply = self._replies.popleft()
        else:
            raise PermanentLLMError("ScriptedLLM has no reply queued")
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(text=reply, usage=LLMUsage(input_tokens=100, output_tokens=50))
