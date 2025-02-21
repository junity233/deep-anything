from deepanything.Stream import Stream,AsyncStream
from openai.types.chat import chat_completion
import openai

from deepanything import Utility

class ResponseClient:
    def __init__(self):
        pass

    def chat_completions(self,messages,model,stream = False,**kwargs) -> Stream or chat_completion.ChatCompletion:
        if stream:
            return self.chat_completions_stream(messages,model,**kwargs)

        return Utility.merge_chunk(self.chat_completions_stream(messages,model,**kwargs),model)

    def chat_completions_stream(self,messages,model,**kwargs) -> Stream:
        pass

class AsyncResponseClient:
    def __init__(self):
        pass

    async def chat_completions(self,messages,model,stream = False,**kwargs) -> AsyncStream or chat_completion.ChatCompletion:
        if stream:
            return self.chat_completions_stream(messages,model,**kwargs)

        return await Utility.async_merge_chunk(await self.chat_completions_stream(messages,model,**kwargs),model)

    async def chat_completions_stream(self,messages,model,**kwargs) -> AsyncStream:
        pass


class OpenaiResponseClient(ResponseClient):
    client : openai.OpenAI

    def __init__(self,base_url,api_key,**kwargs):
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
    client : openai.AsyncOpenAI

    def __init__(self,base_url,api_key,**kwargs):
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