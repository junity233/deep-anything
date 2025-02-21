import time
from collections.abc import Iterable
from typing import Optional, List, Literal, AsyncIterable
import uuid

from openai.types.chat import chat_completion_chunk,chat_completion,chat_completion_message
from openai.types import completion_usage,completion_choice

def make_usage(
        completion_tokens,
        prompt_tokens,
        total_tokens
) -> completion_usage.CompletionUsage:
    usage = completion_usage.CompletionUsage(
        completion_tokens = completion_tokens,
        prompt_tokens = prompt_tokens,
        total_tokens = total_tokens
    )

    return usage

def make_chat_completion_chunk_delta(
        role : Literal["developer", "system", "user", "assistant", "tool"],
        content : Optional[str] = None,
        reasoning_content : Optional[str] = None
) -> chat_completion_chunk.ChoiceDelta:
    delta = chat_completion_chunk.ChoiceDelta(
        role = role,
        content = content
    )

    delta.reasoning_content = reasoning_content

    return delta

def make_chat_completion_chunk_choice(
        delta : chat_completion_chunk.ChoiceDelta,
        index : int = 0,
        finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None,
) -> chat_completion_chunk.Choice:
    choice = chat_completion_chunk.Choice(
        delta = delta,
        index = index,
        finish_reason = finish_reason
    )

    return choice


def make_chat_completion_chunk(
        _id : str,
        usage : Optional[chat_completion_chunk.CompletionUsage],
        model : str,
        choices : List[completion_choice.CompletionChoice],
        created : int = int(time.time()),
) -> chat_completion_chunk.ChatCompletionChunk:
    chunk = chat_completion_chunk.ChatCompletionChunk(
        id = _id,
        created = created,
        model = model,
        choices = choices,
        object = "chat.completion.chunk",
        usage = usage
    )

    return chunk


def make_chat_completion_message(
    role : Literal["developer", "system", "user", "assistant", "tool"],
    content : Optional[str] = None,
    reasoning_content : Optional[str] = None
) -> chat_completion_message.ChatCompletionMessage:
    message = chat_completion_message.ChatCompletionMessage(
        role = role
    )

    message.content = content
    message.reasoning_content = reasoning_content

    return message

def make_chat_completion_choice(
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"],
    message :chat_completion_message.ChatCompletionMessage,
    index : int = 0
) -> chat_completion.Choice:
    choice = chat_completion.Choice(
        message = message,
        finish_reason = finish_reason,
        index = index,
        logprobs = None
    )
    return choice


def make_chat_completion(
        _id : str,
        choices : List[chat_completion.Choice],
        model : str,
        usage : Optional[chat_completion.CompletionUsage],
        created : int = int(time.time()),
) -> chat_completion.ChatCompletion:
    res = chat_completion.ChatCompletion(
        object="chat.completion",
        id = _id,
        choices = choices,
        model = model,
        usage = usage,
        created = created,
        system_fingerprint = None
    )

    return res


def _update_accumulators(
        reasoning: str,
        content: str,
        comp_tokens: int,
        prompt_tokens: int,
        total_tokens: int,
        chunk: chat_completion_chunk.ChatCompletionChunk
) -> tuple[str, str, int, int, int]:
    delta = chunk.choices[0].delta

    if delta.content is None:
        reasoning += delta.reasoning_content or ""
    else:
        content += delta.content or ""

    if chunk.usage is not None:
        comp_tokens,prompt_tokens,total_tokens = chunk.usage.completion_tokens,chunk.usage.prompt_tokens,chunk.usage.total_tokens

    return reasoning, content, comp_tokens, prompt_tokens, total_tokens


def _build_final_completion(
        model: str,
        created: int,
        reasoning_content: str,
        content: str,
        comp_tokens: int,
        prompt_tokens: int,
        total_tokens: int
) -> chat_completion.ChatCompletion:

    usage = make_usage(comp_tokens, prompt_tokens, total_tokens)

    return make_chat_completion(
        _id=str(uuid.uuid4()),
        choices=[make_chat_completion_choice(
            finish_reason="stop",
            message=make_chat_completion_message(
                role="assistant",
                reasoning_content=reasoning_content if reasoning_content else None,
                content=content if content else None
            )
        )],
        model=model,
        usage=usage,
        created=created
    )


def merge_chunk(
        chunks: Iterable[chat_completion_chunk.ChatCompletionChunk],
        model: str,
        created: int = int(time.time())
) -> chat_completion.ChatCompletion:
    """同步合并 chunk 流"""
    reasoning_content = ""
    content = ""
    comp_tokens = prompt_tokens = total_tokens = 0

    for chunk in chunks:
        reasoning_content, content, comp_tokens, prompt_tokens, total_tokens = \
            _update_accumulators(reasoning_content, content, comp_tokens, prompt_tokens, total_tokens, chunk)

    return _build_final_completion(model, created, reasoning_content, content, comp_tokens, prompt_tokens, total_tokens)


async def async_merge_chunk(
        chunks: AsyncIterable[chat_completion_chunk.ChatCompletionChunk],
        model: str,
        created: int = int(time.time())
) -> chat_completion.ChatCompletion:
    """异步合并 chunk 流"""
    reasoning_content = ""
    content = ""
    comp_tokens = prompt_tokens = total_tokens = 0

    async for chunk in chunks:
        reasoning_content, content, comp_tokens, prompt_tokens, total_tokens = \
            _update_accumulators(reasoning_content, content, comp_tokens, prompt_tokens, total_tokens, chunk)

    return _build_final_completion(model, created, reasoning_content, content, comp_tokens, prompt_tokens, total_tokens)

def merge_usage(
        a:completion_usage.CompletionUsage,
        b:completion_usage.CompletionUsage
) -> completion_usage.CompletionUsage:
    return make_usage(
        completion_tokens=a.completion_tokens + b.completion_tokens,
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        total_tokens=a.total_tokens + b.total_tokens
    )

def make_id_by_timestamp():
    return "chatcmpl-" + str(uuid.uuid4())