# DeepAnything

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[中文版](README_zh.md)

DeepAnything is a project that provides DeepSeek R1's deep thinking capabilities for various large language models (LLMs).

## Key Features

- Provides an interface similar to OpenAI
- Supports using various models compatible with the OpenAI API as response models and thinking models
- Offers a server that provides a simple configuration to obtain an API interface compatible with the OpenAI API, and can be called using the official OpenAI SDK.
- Supports using QWQ-32b as a thinking model

## Installation Guide

Install via pip:

```bash
pip install deepanything
```

## Quick Start

### 1. Integrate into Code

#### Chat Completion

```python
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

completions = da_client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": "你好"
        }
    ],
    reason_model="Pro/deepseek-ai/DeepSeek-R1",
    response_model="qwen-max-latest",
    show_model="R1-qwen-max"
)
```

#### Streaming Call

```python
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
```

#### Asynchronous Usage

```python
from deepanything.ReasonClient import AsyncDeepseekReasonClient
from deepanything.ResponseClient import AsyncOpenaiResponseClient
from deepanything.DeepAnythingClient import AsyncDeepAnythingClient
import asyncio

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
    completions = await da_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": "你好"
            }
        ],
        reason_model="Pro/deepseek-ai/DeepSeek-R1",
        response_model="qwen-max-latest",
        show_model="R1-qwen-max"
    )
    print(completions)

asyncio.run(main())
```

More example can be find in [examples](examples).

### 2. Use as a Server

```bash
 python -m deepanything --host host --port port --config config.json
```

| Parameter | Description             |
| --- |----------------|
| --host | Server listening address, will override the setting in config.json |
| --port | Server listening port, will override the setting in config.json |
| --config | Configuration file path |

#### Configuration File Format

Below is an example of a configuration file:

```json
// Using R1 with Qwen-Max-Latest
{
  "host" : "0.0.0.0",
  "port" : 8080,
  "reason_clients": [
    {
      "name" : "siliconflow",
      "type" : "deepseek",
      "base_url" : "https://api.siliconflow.cn/v1",
      "api_key" : "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    }
  ],
  "response_clients": [
    {
      "name" : "qwen",
      "type" : "openai",
      "base_url" : "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key" : "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    }
  ],
  "models": [
    {
      "name": "R1-Qwen-max",
      "reason_client" : "siliconflow",
      "response_client" : "qwen",
      "reason_model": "Pro/deepseek-ai/DeepSeek-R1",
      "response_model" :  "qwen-max-latest"
    }
  ],
  "api_keys" : [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ],
  "log": {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": null
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": "%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr"
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": false},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": false}
    }
  }
}
```

#### **Detailed Explanation**

- reason_clients: Configuration for thinking models, currently supports deepseek and openai types. When the type is openai, deepanything directly uses the model's output as the thinking content, and it is recommended to use qwq-32b in this case.
- response_clients: Configuration for response models, currently only supports the openai type.
- api_keys: API keys for user authentication. When left blank or an empty list, the server does not use API keys for authentication.
- log: Log configuration. If this item is not filled in, the default logging configuration of uvicorn will be used. For details, please refer to [uvicorn logging configuration](https://www.uvicorn.org/settings/#logging).

## Deploying with Docker
### 1. Pull the Image
```bash
docker pull junity233/deepanything:latest
```

### 2. Create config.json
First, create a folder in your desired directory, and then create a file named `config.json` inside it. This folder will be mounted into the container:
```bash
mkdir deepanything-data               # You can replace this with another name
vim deepanything-data/config.json     # Edit the configuration file, you can refer to examples/config.json
```

### 3. Run the Container
```bash
# Remember to modify the port mapping
docker run -v ./deepanything-data:/data -p 8080:8080 junity233/deepanything:latest
```

## License

This project is licensed under the [MIT License](LICENSE)

## Contact Us

Email: 1737636624@qq.com

GitHub Issues: [https://github.com/junity233/deep-anything/issues](https://github.com/junity233/deep-anything/issues)
