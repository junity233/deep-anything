import time
from typing import Optional, List

from openai.types.chat.chat_completion import ChatCompletion

from deepanything.ReasonClient import ReasonClient, AsyncReasonClient
from deepanything.ResponseClient import ResponseClient, AsyncResponseClient
from deepanything.Stream import Stream, AsyncStream
from deepanything.Utility import make_usage, make_chat_completion_message, make_chat_completion, \
    make_chat_completion_choice, merge_usage, make_id_by_timestamp, \
    extend_message


def _merge_chat_completion(
        reason: ChatCompletion,
        response: ChatCompletion,
        show_model: str,
        created: int = int(time.time()),
        _id: str = make_id_by_timestamp()
) -> ChatCompletion:
    return make_chat_completion(
        _id=_id,
        choices=[
            make_chat_completion_choice(
                finish_reason=response.choices[0].finish_reason,
                message=make_chat_completion_message(
                    role="assistant",
                    content=response.choices[0].message.content,
                    reasoning_content=reason.choices[0].message.reasoning_content
                )
            )
        ],
        model=show_model,
        usage=merge_usage(
            reason.usage,
            response.usage
        ),
        created=created
    )

def _build_message(
        messages : List,
        reason_content : str,
        reason_prompt : str
) -> List:
    return extend_message(messages, role="assistant", content=reason_prompt.format(reason_content))
def _process_reason_chunk(chunk, reasoning_contents, reason_usage, show_model, created, _id):
    new_chunk = chunk.model_copy(deep=False)
    new_chunk.model = show_model
    new_chunk.created = created
    new_chunk.id = _id

    if new_chunk.usage is not None:
        reason_usage = new_chunk.usage

    if chunk.choices:
        delta = chunk.choices[0].delta
        reasoning_contents.append(delta.reasoning_content)

    return new_chunk, reason_usage

def _process_response_chunk(chunk, reason_usage,show_model, created, _id):
    new_chunk = chunk.model_copy(deep=False)
    new_chunk.model = show_model
    new_chunk.created = created
    new_chunk.id = _id
    if new_chunk.usage is not None:
        new_usage = new_chunk.usage.model_copy()
        new_usage.completion_tokens += reason_usage.completion_tokens
        new_usage.prompt_tokens += reason_usage.prompt_tokens
        new_usage.total_tokens += reason_usage.total_tokens

        new_chunk.usage = new_usage
    return new_chunk
def chat_completion(
        messages: list,
        reason_client : ReasonClient,
        reason_model: str,
        response_client : ResponseClient,
        response_model: str,
        show_model: str,
        reason_args=None,
        response_args=None,
        reason_prompt: str = "<think>{}</think>",
        reason_system_prompt: Optional[str] = None,
        created: int = int(time.time()),
        stream = False,
        _id: str = make_id_by_timestamp(),
        max_tokens : Optional[int] = None
    ) -> Stream or ChatCompletion:

    """
    Make chat completion synchronously.

    :param messages: Messages
    :param reason_client: Reason client
    :param reason_model: Reason model
    :param response_client: Response client
    :param response_model: Response model
    :param show_model: Specify the model name in the return value
    :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
    :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
    :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
    :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
    :param created: The timestamp indicating the time when the chat completion was created
    :param stream: Whether you use streaming return
    :param _id: Specify the `id` in the return value
    :param max_tokens: max_tokens
    :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
    """

    if response_args is None:
        response_args = {}
    if reason_args is None:
        reason_args = {}
    if stream:
        return chat_completion_stream(
            messages=messages,
            reason_model=reason_model,
            reason_client=reason_client,
            response_model=response_model,
            response_client=response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            created=created,
            _id=_id,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            max_tokens=max_tokens,
        )

    if max_tokens is not None:
        reason_args["max_tokens"] = max_tokens

    reason_chat_completion: ChatCompletion = reason_client.reason(
        messages=messages,
        model=reason_model,
        max_tokens = max_tokens,
        **reason_args
    )

    if max_tokens is not None:
        max_tokens -= reason_chat_completion.usage.completion_tokens
        if max_tokens <= 0:
            return reason_chat_completion
        response_args["max_tokens"] = max_tokens

    response_chat_completion: ChatCompletion = response_client.chat_completions(
        messages=messages + [make_chat_completion_message(
            role="assistant",
            content=reason_prompt.format(reason_chat_completion.choices[0].message.reasoning_content)
        )],
        model=response_model,
        max_tokens = max_tokens,
        **response_args
    )

    return _merge_chat_completion(reason_chat_completion, response_chat_completion, show_model, created, _id)
def chat_completion_stream(
        messages: list,
        reason_client : ReasonClient,
        reason_model: str,
        response_client : ResponseClient,
        response_model: str,
        show_model: str,
        reason_args=None,
        response_args=None,
        reason_prompt: str = "<think>{}</think>",
        reason_system_prompt: Optional[str] = None,
        created: int = int(time.time()),
        _id: str = make_id_by_timestamp(),
        max_tokens : Optional[int] = None
) -> Stream:

    """
    Make chat completion synchronously.This method uses streaming return

    :param messages: Messages
    :param reason_client: Reason client
    :param reason_model: Reason model
    :param response_client: Response client
    :param response_model: Response model
    :param show_model: Specify the model name in the return value
    :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
    :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
    :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
    :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
    :param created: The timestamp indicating the time when the chat completion was created
    :param _id: Specify the `id` in the return value
    :param max_tokens: max_tokens
    :return: A stream
    """

    if response_args is None:
        response_args = {}
    if reason_args is None:
        reason_args = {}
    stream: Optional[Stream] = None

    def _iter():
        nonlocal stream,max_tokens

        # reasoning
        reasoning_contents = []
        if max_tokens is not None:
            reason_args["max_tokens"] = max_tokens

        reason_stream = reason_client.reason_stream(
            messages,
            reason_model,
            reason_system_prompt,
            **reason_args
        )
        stream = reason_stream
        reason_usage = make_usage(0, 0, 0)

        for chunk in reason_stream:
            new_chunk, reason_usage = _process_reason_chunk(chunk, reasoning_contents, reason_usage, show_model, created, _id)
            yield new_chunk

        if max_tokens is not None:
            max_tokens -= reason_usage.completion_tokens
            if max_tokens <= 0:
                return
            response_args["max_tokens"] = max_tokens

        new_messages = _build_message(messages,reason_content="".join(reasoning_contents),reason_prompt=reason_prompt)

        response_stream = response_client.chat_completions_stream(
            new_messages,
            response_model,
            **response_args
        )

        stream = response_stream

        for chunk in response_stream:
            yield _process_response_chunk(chunk, reason_usage,show_model,created,_id)

    return Stream(_iter()).on_next(lambda it: it.__next__()).on_close(lambda _: stream.close())

async def chat_completion_async(
        messages: list,
        reason_client : AsyncReasonClient,
        reason_model: str,
        response_client : AsyncResponseClient,
        response_model: str,
        show_model: str,
        reason_args=None,
        response_args=None,
        reason_prompt: str = "<think>{}</think>",
        reason_system_prompt: Optional[str] = None,
        created: int = int(time.time()),
        _id: str = make_id_by_timestamp(),
        stream=False,
        max_tokens : Optional[int] = None
):
    """
    Make chat completion asynchronously.

    :param messages: Messages
    :param reason_client: Reason client
    :param reason_model: Reason model
    :param response_client: Response client
    :param response_model: Response model
    :param show_model: Specify the model name in the return value
    :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
    :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
    :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
    :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
    :param created: The timestamp indicating the time when the chat completion was created
    :param stream: Whether you use streaming return
    :param _id: Specify the `id` in the return value
    :param max_tokens: max_tokens
    :return: Return a AsyncStream if stream is Ture,otherwise return a ChatCompletion
    """

    if response_args is None:
        response_args = {}
    if reason_args is None:
        reason_args = {}
    if stream:
        return await chat_completion_stream_async(
            messages=messages,
            reason_model=reason_model,
            reason_client=reason_client,
            response_model=response_model,
            response_client=response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            created=created,
            _id=_id,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            max_tokens=max_tokens
        )

    if max_tokens is not None:
        reason_args["max_tokens"] = max_tokens

    reason_chat_completion:ChatCompletion = await reason_client.reason(
        messages=messages,
        model=reason_model,
        **reason_args
    )

    if max_tokens is not None:
        max_tokens -= reason_chat_completion.usage.completion_tokens
        if max_tokens <= 0:
            return reason_chat_completion
        response_args["max_tokens"] = max_tokens

    messages = _build_message(
        messages,
        reason_chat_completion.choices[0].message.reasoning_content,
        reason_prompt
    )

    response_chat_completion:ChatCompletion = await response_client.chat_completions(
        messages=messages,
        model=response_model,
        **response_args
    )

    return _merge_chat_completion(reason_chat_completion,response_chat_completion,show_model,created,_id)

async def chat_completion_stream_async(
        messages: list,
        reason_client : AsyncReasonClient,
        reason_model: str,
        response_client : AsyncResponseClient,
        response_model: str,
        show_model: str,
        reason_args=None,
        response_args=None,
        reason_prompt: str = "<think>{}</think>",
        reason_system_prompt: Optional[str] = None,
        created: int = int(time.time()),
        _id: str = make_id_by_timestamp(),
        max_tokens : Optional[int] = None
) -> AsyncStream:
    """
    Make chat completion asynchronously.This method uses streaming return

    :param messages: Messages
    :param reason_client: Reason client
    :param reason_model: Reason model
    :param response_client: Response client
    :param response_model: Response model
    :param show_model: Specify the model name in the return value
    :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
    :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
    :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
    :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
    :param created: The timestamp indicating the time when the chat completion was created
    :param _id: Specify the `id` in the return value
    :param max_tokens: max_tokens
    :return: A AsyncStream
    """

    if response_args is None:
        response_args = {}
    if reason_args is None:
        reason_args = {}
    stream : Optional[AsyncStream] = None
    async def _iter():
        nonlocal stream,max_tokens

        # reasoning
        reasoning_contents = []
        if max_tokens is not None:
            reason_args["max_tokens"] = max_tokens

        reason_stream = await reason_client.reason_stream(
            messages,
            reason_model,
            reason_system_prompt,
            **reason_args
        )

        stream = reason_stream
        reason_usage = make_usage(0,0,0)

        async for chunk in reason_stream:
            new_chunk,reason_usage = _process_reason_chunk(chunk, reasoning_contents, reason_usage, show_model, created, _id)
            yield new_chunk

        new_messages = _build_message(messages,reason_content="".join(reasoning_contents),reason_prompt=reason_prompt)

        if max_tokens is not None:
            max_tokens -= reason_usage.completion_tokens
            if max_tokens <= 0:
                return
            response_args["max_tokens"] = max_tokens

        response_stream = await response_client.chat_completions_stream(
            new_messages,
            response_model,
            **response_args
        )

        stream = response_stream

        async for chunk in response_stream:
            yield _process_response_chunk(chunk, reason_usage,show_model, created, _id)

    return AsyncStream(_iter()).on_next(lambda it:it.__anext__()).on_close(lambda _:stream.close())

class DeepAnythingClient:
    """
    DeepAnything Client
    """

    reason_client : ReasonClient
    response_client : ResponseClient
    reason_prompt : Optional[str]
    reason_system_prompt : Optional[str]

    def __init__(
            self,
            reason_client: ReasonClient,
            response_client: ResponseClient,
            reason_prompt : str = None,
            reason_system_prompt: Optional[str] = None
    ):
        """
        :param reason_client: Reason client
        :param response_client: Response client
        :param reason_prompt: Default value for reason_prompt
        :param reason_system_prompt: Default value for reason_system_prompt
        """
        self.reason_client = reason_client
        self.response_client = response_client
        self.reason_prompt = reason_prompt
        self.reason_system_prompt = reason_system_prompt

    def chat_completion(
            self,
            messages: list,
            reason_model: str,
            response_model: str,
            show_model : str,
            reason_args=None,
            response_args=None,
            reason_prompt: Optional[str] = None,
            reason_system_prompt: Optional[str] = None,
            created : int = int(time.time()),
            _id : str = make_id_by_timestamp(),
            stream = False,
            max_tokens : Optional[int] = None
    ) -> Stream or ChatCompletion:
        """
        Make chat completion synchronously.

        :param messages: Messages
        :param reason_model: Reason model
        :param response_model: Response model
        :param show_model: Specify the model name in the return value
        :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param created: The timestamp indicating the time when the chat completion was created
        :param stream: Whether you use streaming return
        :param _id: Specify the `id` in the return value
        :param max_tokens: max_tokens
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """
        if reason_prompt is None:
            reason_prompt = self.reason_prompt
        if reason_system_prompt is None:
            reason_system_prompt = self.reason_system_prompt

        return chat_completion(
            messages=messages,
            reason_model=reason_model,
            reason_client=self.reason_client,
            response_model=response_model,
            response_client=self.response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            created=created,
            _id=_id,
            stream=stream,
            max_tokens=max_tokens
        )

    def chat_completion_stream(
            self,
            messages: list,
            reason_model: str,
            response_model: str,
            show_model : str,
            reason_args=None,
            response_args=None,
            reason_prompt: Optional[str] = None,
            reason_system_prompt: Optional[str] = None,
            created : int = int(time.time()),
            _id : str = make_id_by_timestamp(),
            max_tokens : Optional[int] = None
    ) -> Stream:
        """
        Make chat completion synchronously.This method uses streaming return

        :param messages: Messages
        :param reason_model: Reason model
        :param response_model: Response model
        :param show_model: Specify the model name in the return value
        :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
        :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
        :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
        :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
        :param created: The timestamp indicating the time when the chat completion was created
        :param _id: Specify the `id` in the return value
        :param max_tokens: max_tokens
        :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """

        if reason_prompt is None:
            reason_prompt = self.reason_prompt
        if reason_system_prompt is None:
            reason_system_prompt = self.reason_system_prompt
        return chat_completion_stream(
            messages=messages,
            reason_model=reason_model,
            reason_client=self.reason_client,
            response_model=response_model,
            response_client=self.response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            created=created,
            _id=_id,
            max_tokens=max_tokens
        )


class AsyncDeepAnythingClient:
    """
        DeepAnything Async Client
    """
    reason_client : AsyncReasonClient
    response_client : AsyncResponseClient
    reason_prompt : Optional[str]
    reason_system_prompt : Optional[str]

    def __init__(
            self,
            reason_client: AsyncReasonClient,
            response_client: AsyncResponseClient,
            reason_prompt : Optional[str] = None,
            reason_system_prompt: Optional[str] = None
    ):
        """
        :param reason_client: Reason client
        :param response_client: Response client
        :param reason_prompt: Default value for reason_prompt
        :param reason_system_prompt: Default value for reason_system_prompt
        """
        self.reason_client = reason_client
        self.response_client = response_client
        self.reason_prompt = reason_prompt
        self.reason_system_prompt = reason_system_prompt

    async def chat_completion(
            self,
            messages: list,
            reason_model: str,
            response_model: str,
            show_model: str,
            reason_args=None,
            response_args=None,
            reason_prompt: Optional[str] = None,
            reason_system_prompt: Optional[str] = None,
            created: int = int(time.time()),
            _id: str = make_id_by_timestamp(),
            stream=False,
            max_tokens : Optional[int] = None
    ):
        """
            Make chat completion asynchronously.

            :param stream: Whether you use streaming return
            :param messages: Messages
            :param reason_model: Reason model
            :param response_model: Response model
            :param show_model: Specify the model name in the return value
            :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
            :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
            :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
            :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
            :param created: The timestamp indicating the time when the chat completion was created
            :param _id: Specify the `id` in the return value
            :param max_tokens: max_tokens
            :return: Return an AsyncStream if stream is Ture,otherwise return a ChatCompletion
        """

        if reason_prompt is None:
            reason_prompt = self.reason_prompt
        if reason_system_prompt is None:
            reason_system_prompt = self.reason_system_prompt

        return await chat_completion_async(
            messages=messages,
            reason_model=reason_model,
            reason_client=self.reason_client,
            response_model=response_model,
            response_client=self.response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            created=created,
            _id=_id,
            stream=stream,
            max_tokens=max_tokens
        )

    async def chat_completion_stream(
            self,
            messages: list,
            reason_model: str,
            response_model: str,
            show_model : str,
            reason_args=None,
            response_args=None,
            reason_prompt: Optional[str] = None,
            reason_system_prompt: Optional[str] = None,
            created : int = int(time.time()),
            _id : str = make_id_by_timestamp(),
            max_tokens : Optional[int] = None
    ) -> AsyncStream:
        """
            Make chat completion asynchronously.This method uses streaming return

            :param messages: Messages
            :param reason_model: Reason model
            :param response_model: Response model
            :param show_model: Specify the model name in the return value
            :param reason_args: Additional parameters passed to the reason client, such as temperature, top_k, etc.
            :param response_args: Additional parameters passed to the response client, such as temperature, top_k, etc.
            :param reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
            :param reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
            :param created: The timestamp indicating the time when the chat completion was created
            :param _id: Specify the `id` in the return value
            :param max_tokens: max_tokens
            :return: Return a Stream if stream is Ture,otherwise return a ChatCompletion
        """

        if reason_prompt is None:
            reason_prompt = self.reason_prompt
        if reason_system_prompt is None:
            reason_system_prompt = self.reason_system_prompt

        return await chat_completion_stream_async(
            messages=messages,
            reason_model=reason_model,
            reason_client=self.reason_client,
            response_model=response_model,
            response_client=self.response_client,
            show_model=show_model,
            reason_args=reason_args,
            response_args=response_args,
            reason_prompt=reason_prompt,
            reason_system_prompt=reason_system_prompt,
            created=created,
            _id=_id,
            max_tokens=max_tokens
        )