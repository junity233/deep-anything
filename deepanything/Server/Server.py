from dataclasses import dataclass
import time
import uvicorn
from typing import Dict, List, Optional, Any
import json

from openai.types.model import Model as OpenaiModel
from fastapi import FastAPI,Depends, HTTPException, status,Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from deepanything.DeepAnythingClient import chat_completion_stream_async, chat_completion_async
from deepanything.ResponseClient import AsyncOpenaiResponseClient,AsyncResponseClient
from deepanything.Stream import AsyncStream
from deepanything.ReasonClient import AsyncDeepseekReasonClient,AsyncOpenaiReasonClient,AsyncReasonClient
from deepanything.Server import Types

@dataclass
class ModelInfo:
    name : str
    reason_client : str
    reason_model : str
    response_client : str
    response_model : str
    created : int = int(time.time())
    reason_prompt : str = "<Think>{}</Think>"

class DeepAnythingServer:
    app : FastAPI = FastAPI()
    host : str
    port : int
    reason_clients : Dict[str,AsyncReasonClient] = {}
    response_clients : Dict[str,AsyncResponseClient] = {}
    models : Dict[str,ModelInfo] = {}
    model_owner : str = "deepanything"
    api_keys : List[str] = []
    security = HTTPBearer()

    def __init__(self, host:str = None, port:int = None, config : Any or str = None):
        if config is not None:
            if isinstance(config,str):
                with open(config) as f:
                    config = json.load(f)
            self.load_config(config)

        if host:
            self.host = host
        if port:
            self.port = port

        self.app.add_api_route("/v1/models",self.get_models,methods=["GET"],response_model=Types.ModelsListResponse)
        self.app.add_api_route("/v1/chat/completions",self.chat_completions,methods=["POST"])

    def run(self):
        uvicorn.run(self.app,host=self.host,port=self.port,log_level="trace")

    @staticmethod
    def _extract_args(query : Types.ChatCompletionQuery) -> dict:
        args = query.dict()
        for key in ["messages","model","stream"]:
            args.pop(key)
        return args

    def load_config(self,config_object : Dict) -> None:
        self.host = config_object.get("host","0.0.0.0")
        self.port = config_object.get("port",8000)
        self.model_owner = config_object.get("model_owner","deepanything")

        reason_clients:List[Dict] = config_object.get("reason_clients",[])

        for client in reason_clients:
            name = client["name"]
            base_url = client["base_url"]
            api_key = client.get("api_key","")
            extract_args = client.get("extract_args",{})

            if client["type"] == 'deepseek':
                self.reason_clients[name] = AsyncDeepseekReasonClient(base_url, api_key, **extract_args)
            elif client["type"] == 'openai':
                self.reason_clients[name] = AsyncOpenaiReasonClient(base_url, api_key, **extract_args)
            else:
                raise Exception("unknown reason client type")

        response_clients : List[Dict] = config_object.get("response_clients",[])

        for client in response_clients:
            name = client["name"]
            base_url = client["base_url"]
            api_key = client.get("api_key","")
            extract_args = client.get("extract_args",{})

            if client["type"] == 'openai':
                self.response_clients[name] = AsyncOpenaiResponseClient(base_url,api_key,**extract_args)
            else:
                raise Exception("unknown response client type")

        models : List[Dict] = config_object.get("models",[])

        for _model in models:
            name = _model["name"]
            reason_client = _model["reason_client"]
            reason_model = _model["reason_model"]
            response_client = _model["response_client"]
            response_model = _model["response_model"]
            created = _model.get("created", int(time.time()))
            reason_prompt = _model.get("reason_prompt","<Think>{}</Think>")

            self.models[name] = ModelInfo(
                name = name,
                reason_client = reason_client,
                reason_model = reason_model,
                response_client = response_client,
                response_model = response_model,
                created = created,
                reason_prompt = reason_prompt
            )

        self.api_keys = config_object.get("api_keys",[])


    def add_reason_client(self,name:str,client:AsyncReasonClient):
        self.reason_clients[name] = client

    def add_response_client(self,name:str,client:AsyncResponseClient):
        self.response_clients[name] = client

    def add_model(self,name:str,model:ModelInfo):
        self.models[name] = model
    def verify_authorization(self, authorization:Optional[str]):
        if not self.api_keys:
            return

        if authorization is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Expect token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token  =authorization[7:]
        if token not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def chat_completions(self, query : Types.ChatCompletionQuery, authorization:Optional[str] = Header(None)):
        self.verify_authorization(authorization)
        model = self.models[query.model]
        async def sse(it: AsyncStream):
            async for chunk in it:
                yield f"data: {chunk.model_dump_json(indent=None)}\n\n"
            yield "data: [DONE]"

        args = DeepAnythingServer._extract_args(query)

        max_tokens = None
        if "max_tokens" in args:
            max_tokens = args["max_tokens"]
            args.pop("max_tokens")

        if query.stream:
            res = sse(
                await chat_completion_stream_async(
                    messages=query.messages,
                    reason_client=self.reason_clients[model.reason_client],
                    reason_model=model.reason_model,
                    response_client=self.response_clients[model.response_client],
                    response_model=model.response_model,
                    show_model=model.name,
                    reason_prompt=model.reason_prompt,
                    response_args=args,
                    reason_args=args,
                    max_tokens=max_tokens
                )
            )
            return StreamingResponse(
                res,
                media_type="text/event-stream"
            )
        else:
            res = await chat_completion_async(
                    messages=query.messages,
                    reason_client=self.reason_clients[model.reason_client],
                    reason_model=model.reason_model,
                    response_client=self.response_clients[model.response_client],
                    response_model=model.response_model,
                    show_model=model.name,
                    reason_prompt=model.reason_prompt,
                    response_args=args,
                    reason_args=args,
                    max_tokens=max_tokens
            )
            return res.model_dump_json()

    def get_models(self) -> Types.ModelsListResponse:
        return Types.ModelsListResponse(
            data = [OpenaiModel(
                    id = model_info.name,
                    owned_by = self.model_owner,
                    created = model_info.created,
                    object = "model"
                )
                for model_info in self.models.values()
            ]
        )