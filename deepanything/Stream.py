from collections.abc import Callable, Awaitable
from typing import Any,  AsyncIterator,  Iterator
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

class Stream:
    """
    Implementation of streaming return. Implement using simple callback functions
    """
    next_fc : Callable[[Any],ChatCompletionChunk]
    close_fc : Callable[[Any],None]
    data : Any


    def __init__(self,data):
        self.data = data
    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def on_next(self,fc : Callable[[Any],ChatCompletionChunk]) -> 'Stream':
        """
        Set callback for `__next__()`

        :param fc: Callback
        :return: Stream itself.
        """
        self.next_fc = fc
        return self

    def on_close(self,fc : Callable[[Any],None]) -> 'Stream':
        """
        Set callback for `close()`

        :param fc: Callback
        :return: Stream itself.
        """
        self.close_fc = fc
        return self

    def __next__(self) -> ChatCompletionChunk:
        return self.next_fc(self.data)

    def close(self) -> None:
        """
        Close the stream
        :return: None
        """
        self.close_fc(self.data)

class AsyncStream:
    """
    Implementation of streaming return. Implement using simple callback functions
    """
    next_fc: Callable[[Any], Awaitable[ChatCompletionChunk]]
    close_fc: Callable[[Any], Awaitable[None]]
    data : Any

    def __init__(self,data):
        self.data = data

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    def on_next(self, fc: Callable[[Any], Awaitable[ChatCompletionChunk]]) -> 'AsyncStream':
        """
        Set callback for `__anext__()`

        :param fc: Callback
        :return: Stream itself.
        """
        self.next_fc = fc
        return self

    def on_close(self, fc: Callable[[Any], Awaitable[None]]) -> 'AsyncStream':
        """
        Set callback for `close()`

        :param fc: Callback
        :return: Stream itself.
        """
        self.close_fc = fc
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        return await self.next_fc(self.data)

    async def close(self) -> None:
        """
        Close the stream
        :return: None
        """
        await self.close_fc(self.data)