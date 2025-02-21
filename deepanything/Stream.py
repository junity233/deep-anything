from collections.abc import Callable, Awaitable
from typing import Any,  AsyncIterator,  Iterator
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

class Stream:
    next_fc : Callable[[Any],ChatCompletionChunk]
    close_fc : Callable[[Any],None]
    data : dict


    def __init__(self,data):
        self.data = data
    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def on_next(self,fc : Callable[[Any],ChatCompletionChunk]) -> 'Stream':
        self.next_fc = fc
        return self

    def on_close(self,fc : Callable[[Any],None]) -> 'Stream':
        self.close_fc = fc
        return self

    def __next__(self) -> ChatCompletionChunk:
        return self.next_fc(self.data)

    def close(self) -> None:
        self.close_fc(self.data)

class AsyncStream:
    next_fc: Callable[[Any], Awaitable[ChatCompletionChunk]]
    close_fc: Callable[[Any], Awaitable[None]]
    data : Any

    def __init__(self,data):
        self.data = data

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    def on_next(self, fc: Callable[[Any], Awaitable[ChatCompletionChunk]]) -> 'AsyncStream':
        self.next_fc = fc
        return self

    def on_close(self, fc: Callable[[Any], Awaitable[None]]) -> 'AsyncStream':
        self.close_fc = fc
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        return await self.next_fc(self.data)

    async def close(self) -> None:
        await self.close_fc(self.data)