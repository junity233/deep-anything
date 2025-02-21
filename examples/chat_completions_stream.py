from deepanything.ReasonClient import DeepseekReasonClient
from deepanything.ResponseClient import OpenaiResponseClient
from deepanything.DeepAnythingClient import DeepAnythingClient

think_client = DeepseekReasonClient(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxxxxxxxx"
)

response_client = OpenaiResponseClient(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-xxxxxxxxxx",
)

da_client = DeepAnythingClient(
    reason_client=think_client,
    response_client=response_client,
    reason_prompt="<Think>{}</Think>"
)

stream = da_client.chat_completion(
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

for chunk in stream:
    print(chunk)