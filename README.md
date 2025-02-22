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
    },
    {
      "name" : "qwen",
      "type" : "openai",
      "base_url" : "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key" : "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
      "response_model" :  "qwen-max-latest",
      "reason_prompt" : "<think>{}</think>"
    },
    {
      "name": "QWQ-Qwen-max",
      "reason_client" : "qwen",
      "response_client" : "qwen",
      "reason_model": "qwq-32b-preview",
      "response_model" :  "qwen-max-latest",
      "reason_prompt" : "<think>{}</think>",
      "reason_system_prompt" : "You are a model designed to contemplate questions before providing answers. Your thought process and responses are not directly visible to the user but are instead passed as prompts to the next model. For any question posed by the user, you should carefully consider it and provide as detailed a thought process as possible."
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
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": false},
        "deepanything": {"handlers": ["default"], "level": "INFO", "propagate": false}
    }
  }
}
```

#### **Detailed Explanation**

- reason_clients: Configuration for the thinking model, currently supporting two types: deepseek and openai. When the type is openai, DeepAnything directly uses the model's output as the thinking content; it is recommended to use qwq-32b in this case.
- response_clients: Configuration for the response model, currently supporting only one type: openai.
- models: Model configuration, including model name, thinking model, response model, parameters for the thinking model, and parameters for the response model.
    - reason_prompt: Specifies how the thinking content should be embedded into the conversation. DeepAnything will use `reason_prompt` to format the thinking content. The default is `<think>{}</think>`.
    - reason_system_prompt: Adds extra prompt words for the thinking model. This prompt will be placed at the end of the message as a `system` role and passed to the thinking model. If not specified, it will not take effect.
- api_keys: API keys used for user identity verification. When not filled or an empty list, the server does not use API keys for authentication.
- log: Log configuration. If this item is not filled, the default uvicorn log configuration will be used. For more details, refer to [uvicorn logging configuration](https://www.uvicorn.org/settings/#logging) and [Python Logging configuration](https://docs.python.org/3/library/logging.config.html).}

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
