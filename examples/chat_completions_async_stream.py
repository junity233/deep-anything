from deepanything.ReasonClient import AsyncDeepseekReasonClient
from deepanything.ResponseClient import AsyncOpenaiResponseClient
from deepanything.DeepAnythingClient import AsyncDeepAnythingClient

think_client = AsyncDeepseekReasonClient(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxxxxxxxx"
)

response_client = AsyncOpenaiResponseClient(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-xxxxxxxxxx",
)

da_client = AsyncDeepAnythingClient(
    reason_client=think_client,
    response_client=response_client,
    reason_prompt="<Think>{}</Think>"
)

async def main():
    stream = await da_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": "你好"
            }
        ],
        reason_model="Pro/deepseek-ai/DeepSeek-R1",
        response_model="qwen-max-latest",
        show_model="R1-qwen-max",
        stream=True
    )

    async for chunk in stream:
        print(chunk)