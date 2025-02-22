# DeepAnything

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[English](README.md)

DeepAnything是一个为各类大语言模型（LLM）提供DeepSeek R1的深度思考能力的项目。

## 主要特性

- 提供与OpenAI类似的接口
- 支持使用兼容OpenAI API的各类模型作为回复模型和思考模型
- 提供一个服务器，提供简单配置即可获得一个兼容OpenAI API的API接口，并使用openai官方SDK调用。
- 可使用QWQ-32b 作为思考模型

## 安装指南

通过pip安装：
```bash
pip install deepanything
```

## 快速开始

### 一.集成到代码中使用
#### 对话补全

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

#### 流式调用
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

#### 异步使用
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
更多示例可以在 [示例](examples) 中查看。

### 二.作为服务器使用
```bash
 python -m deepanything --host host --port port --config config.json
```
| 参数 | 说明             |
| --- |----------------|
| --host | 服务器监听地址 ，指定后会覆盖config.json中的设置 |
| --port | 服务器监听端口，指定后会覆盖config.json中的设置 |
| --config | 配置文件路径|

#### 配置文件格式
下面是一个配置文件的示例：

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
**详细解释**

- reason_clients：思考模型的配置，目前支持deepseek和openai两种类型。当type为openai时，deepanything直接使用模型的输出作为思考内容，推荐使用qwq-32b在这种情况下使用。
- response_clients：回复模型的配置，目前仅支持openai一种类型。
- api_keys：API密钥，用于验证用户身份。当不填写或为空列表时，服务器不使用API密钥进行身份验证。
- log: 日志配置，若不填写此项则使用uvicorn默认的日志配置。具体可参考 [uvicorn日志配置](https://www.uvicorn.org/settings/#logging)。

## 使用 Docker 部署
### 1.拉取镜像
```bash
docker pull junity233/deepanything:latest
```

### 2.创建 config.json
首先在一个你希望的目录下创建一个文件夹，然后在里面创建一个名为 config.json 的文件。这个文件夹将会被挂载到容器中：
```bash
mkdir deepanything-data               # 可以替换为其它名称
vim deepanything-data/config.json     # 编写配置文件，可以参考 examples/config.json
```

### 3.运行容器
```bash
# 记得修改端口映射
docker run -v ./deepanything-data:/data -p 8080:8080 junity233/deepanything:latest
```

## 许可证
本项目采用 [MIT 许可证](LICENSE)

## 联系我们
邮箱：1737636624@qq.com

GitHub Issues：https://github.com/junity233/deep-anything/issues