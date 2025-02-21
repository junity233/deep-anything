from pydantic import BaseModel
from openai.types.model import Model as OpenaiModel
from typing import Dict,List

class ModelsListResponse(BaseModel):
    data : List[OpenaiModel]

class ChatCompletionMessage(BaseModel):
    role : str
    content : str

class ChatCompletionQuery(BaseModel):
    model : str
    messages : List[ChatCompletionMessage]
    stream : bool = False