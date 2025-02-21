from pydantic import BaseModel
from openai.types.model import Model as OpenaiModel
from typing import Dict, List, Optional


class ModelsListResponse(BaseModel):
    data : List[OpenaiModel]

class ChatCompletionMessage(BaseModel):
    role : str
    content : str

class ChatCompletionQuery(BaseModel):
    model : str
    messages : List[ChatCompletionMessage]
    stream : bool = False
    temperature : Optional[float] = 0.7
    top_p : Optional[float] = 1.0
    frequency_penalty : Optional[float] = 0.0
    presence_penalty : Optional[float] = 0.0
    max_tokens : Optional[int] = 128000
    stop : Optional[List[str]] = None