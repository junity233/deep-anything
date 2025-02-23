from typing import Optional

import openai
from openai.types.chat import chat_completion, chat_completion_chunk
from deepanything.Stream import Stream,AsyncStream
from deepanything import Utility


class ReasonClient:
    """
    Base Class for Reason Clients
    """
    def __init__(self) -> None:
        pass

    def reason(
            self,
            messages:list[dict],
            model:str,
            reason_system_prompt:Optional[str] = None,
            stream = False,
            **kwargs
    ) -> Stream or chat_completion.ChatCompletion:
        """
        Generate reason content like Deepseek R1. This function returns a value that is almost the same as the OpenAI API, but 'content' is None and 'reasoning_content' is reason content.

        :param messages: Messages
        :param model: Model
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param stream: Whether you use streaming return
        :param kwargs: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        if stream:
            return self.reason_stream(messages, model, **kwargs)

        return Utility.merge_chunk(
            self.reason_stream(messages, model, **kwargs),
            model
        )

    def reason_stream(self,
                      messages:list[dict],
                      model:str,
                      reason_system_prompt:Optional[str] = None,
                      **kwargs
                      ) -> Stream:
        """
        Generate reason content like Deepseek R1. This function returns a value that is almost the same as the OpenAI API, but 'content' is None and 'reasoning_content' is reason content.This method uses streaming return

        :param messages: Messages
        :param model: Model
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param kwargs: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        raise NotImplementedError

class AsyncReasonClient:
    """
    Base Class for Async Reason Clients
    """
    def __init__(self) -> None:
        pass

    async def reason(
            self,
            messages:list[dict],
            model:str,
            reason_system_prompt:Optional[str] = None,
            stream = False,
            **kwargs
    ) -> AsyncStream or chat_completion.ChatCompletion:
        """
        Generate reason content like Deepseek R1. This function returns a value that is almost the same as the OpenAI API, but 'content' is None and 'reasoning_content' is reason content.

        :param messages: Messages
        :param model: Model
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param stream: Whether you use streaming return
        :param kwargs: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        if stream:
            return await self.reason_stream(messages, model, **kwargs)

        return await Utility.async_merge_chunk(
            await self.reason_stream(messages, model, **kwargs),
            model
        )

    async def reason_stream(self,
                            messages:list[dict],
                            model:str,
                            reason_system_prompt:Optional[str] = None,
                            **kwargs
                            ) -> AsyncStream:
        """
        Generate reason content like Deepseek R1. This function returns a value that is almost the same as the OpenAI API, but 'content' is None and 'reasoning_content' is reason content.This method uses streaming return

        :param messages: Messages
        :param model: Model
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param kwargs: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :return: Return a AsyncStream if stream is Ture,otherwise return a ChatCompletion
        """
        raise NotImplementedError

class DeepseekReasonClient(ReasonClient):
    """
    Deepseek Reason Client
    """
    client : openai.OpenAI

    def __init__(self,base_url:str,api_key:str,**kwargs) -> None:
        """
        :param base_url: Base url
        :param api_key: Api key
        :param kwargs: Other parameters used to create clients
        """
        super().__init__()
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    def reason_stream(self,
                      messages: list[dict],
                      model: str,
                      reason_system_prompt:Optional[str] = None, # not used
                      **kwargs
                      ) -> Stream:
        stream = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            stream_options = {"include_usage": True},
            **kwargs
        )

        def _iter():
            for chunk in stream:
                if len(chunk.choices) == 0:
                    yield chunk
                if chunk.choices[0].delta.reasoning_content is not None:
                    yield chunk
                else:
                    return


        return (Stream(_iter())
                .on_next(lambda it : it.__next__())
                .on_close(lambda _: stream.close()))

class AsyncDeepseekReasonClient(AsyncReasonClient):
    """
    Deepseek Reason Async Client
    """
    client : openai.AsyncOpenAI

    def __init__(self,base_url:str,api_key:str,**kwargs) -> None:
        """
        :param base_url: Base url
        :param api_key: Api key
        :param kwargs: Other parameters used to create clients
        """

        super().__init__()
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    async def reason_stream(self,
                            messages: list[dict],
                            model: str,
                            reason_system_prompt:Optional[str] = None,
                            **kwargs
                            ) -> AsyncStream:
        stream = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            stream_options = {"include_usage": True},
            **kwargs
        )

        async def _iter():
            async for chunk in stream:
                if len(chunk.choices) == 0:
                    yield chunk
                    continue
                if chunk.choices[0].delta.reasoning_content is not None:
                    yield chunk
                else:
                    return


        return (AsyncStream(_iter())
                .on_next(lambda it : it.__anext__())
                .on_close(lambda _: stream.close()))


def _rebuild_chunk_for_openai(
        chunk:chat_completion_chunk.ChatCompletionChunk
) -> chat_completion_chunk.ChatCompletionChunk:
    if len(chunk.choices):
        chunk.choices[0].delta.reasoning_content = chunk.choices[0].delta.content
        chunk.choices[0].delta.content = None
    return chunk


class OpenaiReasonClient(ReasonClient):
    """
    OpenAI Reason Client.Used When using models similar to QWQ
    """
    client : openai.OpenAI
    def __init__(
            self,
            base_url:str,
            api_key:str,
            **kwargs
    ) -> None:
        """
        :param base_url: Base url
        :param api_key: Api key
        :param kwargs: Other parameters used to create clients
        """
        super().__init__()
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    def reason_stream(self,
                      messages: list[dict],
                      model: str,
                      reason_system_prompt:Optional[str] = None,
                      **kwargs
                      ) -> Stream:
        if reason_system_prompt is not None:
            messages = Utility.extend_message(messages, role="system", content=reason_system_prompt)

        stream =  self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            stream_options = {"include_usage": True},
            **kwargs
        )

        return Stream(stream).on_next(lambda it: _rebuild_chunk_for_openai(it.__next__())).on_close(lambda _: stream.close())

    def reason(
            self,
            messages:list[dict],
            model:str,
            reason_system_prompt:Optional[str] = None,
            stream = False,
            **kwargs
    ) -> Stream or chat_completion.ChatCompletion:
        if stream:
            return self.reason_stream(messages, model, **kwargs)

        if reason_system_prompt is not None:
            messages = Utility.extend_message(messages, role="system", content=reason_system_prompt)

        completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs
        )

        completion.choices[0].message.reasoning_content = completion.choices[0].message.content
        completion.choices[0].message.content = None

        return completion

class AsyncOpenaiReasonClient(AsyncReasonClient):
    """
    OpenAI Reason Async Client.Used When using models similar to QWQ
    """
    client : openai.AsyncOpenAI
    def __init__(self,base_url:str,api_key:str,**kwargs) -> None:
        """
        :param base_url: Base url
        :param api_key: Api key
        :param kwargs: Other parameters used to create clients
        """
        super().__init__()
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    async def reason_stream(self,
                            messages: list[dict],
                            model: str,
                            reason_system_prompt:Optional[str] = None,
                            **kwargs
                            ) -> AsyncStream:

        if reason_system_prompt is not None:
            messages = Utility.extend_message(messages, role="system", content=reason_system_prompt)

        stream =  await self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            stream_options = {"include_usage": True},
            **kwargs
        )

        async def _next(it):
            return _rebuild_chunk_for_openai(await it.__anext__())

        return AsyncStream(stream).on_next(lambda it: _next(it)).on_close(lambda _: stream.close())

    async def reason(self,
                     messages: list[dict],
                     model: str,
                     reason_system_prompt:Optional[str] = None,
                     stream = False,
                     **kwargs
                     ) -> AsyncStream or chat_completion.ChatCompletion:
        if stream:
            return await self.reason_stream(messages, model, **kwargs)

        if reason_system_prompt is not None:
            messages = Utility.extend_message(messages, role="system", content=reason_system_prompt)

        completion = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            **kwargs
        )

        completion.choices[0].message.reasoning_content = completion.choices[0].message.content
        completion.choices[0].message.content = None

        return completion