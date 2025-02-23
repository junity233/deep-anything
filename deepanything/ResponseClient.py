from deepanything.Stream import Stream,AsyncStream
from openai.types.chat import chat_completion
import openai

from deepanything import Utility

class ResponseClient:
    """
    Base Class for Response Client
    """
    def __init__(self):
        pass

    def chat_completions(self,messages,model,stream = False,**kwargs) -> Stream or chat_completion.ChatCompletion:
        """
        Make chat completion for responding

        :param messages: Messages
        :param model: Model
        :param stream: Whether you use streaming return
        :param kwargs: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        if stream:
            return self.chat_completions_stream(messages,model,**kwargs)

        return Utility.merge_chunk(self.chat_completions_stream(messages,model,**kwargs),model)

    def chat_completions_stream(self,messages,model,**kwargs) -> Stream:
        """
        Make chat completion for responding.This method uses streaming return

        :param messages: Messages
        :param model: Model
        :param kwargs: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        raise NotImplementedError()

class AsyncResponseClient:
    """
    Base Class for Response Async Client
    """
    def __init__(self):
        pass

    async def chat_completions(self,messages,model,stream = False,**kwargs) -> AsyncStream or chat_completion.ChatCompletion:
        """
        Make chat completion for responding

        :param messages: Messages
        :param model: Model
        :param stream: Whether you use streaming return
        :param kwargs: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        if stream:
            return self.chat_completions_stream(messages,model,**kwargs)

        return await Utility.async_merge_chunk(await self.chat_completions_stream(messages,model,**kwargs),model)

    async def chat_completions_stream(self,messages,model,**kwargs) -> AsyncStream:
        """
        Make chat completion for responding.This method uses streaming return

        :param messages: Messages
        :param model: Model
        :param kwargs: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        raise NotImplementedError()

class OpenaiResponseClient(ResponseClient):
    """
    OpenAI-like response client
    """
    client : openai.OpenAI

    def __init__(self,base_url,api_key,**kwargs):
        """

        :param base_url:  Base url
        :param api_key: API Key
        :param kwargs: Other parameters used to create clients
        """
        super().__init__()
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    def chat_completions(self,messages,model,stream = False,**kwargs) -> Stream or chat_completion.ChatCompletion:
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs
        )

    def chat_completions_stream(self,messages,model,**kwargs) -> Stream:
        return self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                stream_options = {"include_usage": True},
                **kwargs
            )


class AsyncOpenaiResponseClient(AsyncResponseClient):
    """
    OpenAI-like async response client
    """
    client : openai.AsyncOpenAI

    def __init__(self,base_url,api_key,**kwargs):
        """
        :param base_url:  Base url
        :param api_key: API Key
        :param kwargs: Other parameters used to create clients
        """
        super().__init__()
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    async def chat_completions(self,messages,model,stream = False,**kwargs) -> AsyncStream or chat_completion.ChatCompletion:
        return await self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs
        )

    async def chat_completions_stream(self,messages,model,**kwargs) -> AsyncStream:
        return await self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                stream_options = {"include_usage": True},
                **kwargs
            )