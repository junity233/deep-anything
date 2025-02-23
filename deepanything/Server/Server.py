import json
import logging
import logging.config
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status, Header, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.security import HTTPBearer
from openai.types.model import Model as OpenaiModel
from uvicorn.config import LOGGING_CONFIG

from deepanything.DeepAnythingClient import chat_completion_stream_async, chat_completion_async
from deepanything.ReasonClient import AsyncDeepseekReasonClient, AsyncOpenaiReasonClient, AsyncReasonClient
from deepanything.ResponseClient import AsyncOpenaiResponseClient, AsyncResponseClient
from deepanything.Server import Types
from deepanything.Stream import AsyncStream
from deepanything.metadatas import VERSION, PYTHON_RUNTIME


@dataclass
class ModelInfo:
    name : str
    reason_client : str
    reason_model : str
    response_client : str
    response_model : str
    created : int = int(time.time())
    reason_prompt : str = "<think>{}</think>",
    reason_system_prompt : Optional[str] = None

class DeepAnythingServer:
    app : FastAPI = FastAPI()
    host : str
    port : int
    logger : logging.Logger
    reason_clients : Dict[str,AsyncReasonClient] = {}
    response_clients : Dict[str,AsyncResponseClient] = {}
    models : Dict[str,ModelInfo] = {}
    model_owner : str = "deepanything"
    api_keys : List[str] = []
    security = HTTPBearer()
    log_config : Dict[str,Any] = LOGGING_CONFIG

    def __init__(self, host:str = None, port:int = None, config : Any or str = None):
        print(f"DeepAnything Server {VERSION} on {PYTHON_RUNTIME}")
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
        self.logger.info(f"DeepAnything server is now running at http://{self.host}:{self.port}")
        uvicorn.run(self.app,host=self.host,port=self.port,log_config=self.log_config)

    @staticmethod
    def _extract_args(query : Types.ChatCompletionQuery) -> dict:
        args = query.dict()
        for key in ["messages","model","stream"]:
            args.pop(key)
        return args

    def load_config(self,config_object : Dict) -> None:
        print("Loading config")
        self.host = config_object.get("host","0.0.0.0")
        self.port = config_object.get("port",8000)
        self.model_owner = config_object.get("model_owner","deepanything")

        self.log_config = config_object.get("log",LOGGING_CONFIG)
        if self.log_config == {}:
            self.log_config = LOGGING_CONFIG
        logging.config.dictConfig(self.log_config)
        self.logger = logging.getLogger("deepanything")

        self._load_reason_clients(config_object)
        self._load_response_clients(config_object)
        self._load_models(config_object)

        self.api_keys = config_object.get("api_keys",[])


    def _load_models(self, config_object):
        self.logger.info("Loading models")
        models: List[Dict] = config_object.get("models", [])
        for _model in models:
            name = _model["name"]
            reason_client = _model["reason_client"]
            reason_model = _model["reason_model"]
            response_client = _model["response_client"]
            response_model = _model["response_model"]
            created = _model.get("created", int(time.time()))
            reason_prompt = _model.get("reason_prompt", "<think>{}</think>")
            reason_system_prompt = _model.get("reason_system_prompt", None)

            if name in self.models:
                self.logger.error(f"Detected duplicate model : {name}")

            if reason_client not in self.reason_clients:
                self.logger.error(f"Reason client '{reason_model}' for '{name}' not found")
                exit(0)

            if response_client not in self.response_clients:
                self.logger.error(f"Response client '{response_model}' for '{name}' not found")
                exit(0)

            self.models[name] = ModelInfo(
                name=name,
                reason_client=reason_client,
                reason_model=reason_model,
                response_client=response_client,
                response_model=response_model,
                created=created,
                reason_prompt=reason_prompt,
                reason_system_prompt=reason_system_prompt
            )

            self.logger.info(f"Loaded model : {name}")

    def _load_response_clients(self, config_object):
        self.logger.info("Loading response clients")
        response_clients: List[Dict] = config_object.get("response_clients", [])
        for client in response_clients:
            name = client["name"]
            base_url = client["base_url"]
            api_key = client.get("api_key", "")
            extract_args = client.get("extract_args", {})

            if name in self.response_clients:
                self.logger.error(f"Detected duplicate response clients : {name}")
                exit(0)

            if client["type"] == 'openai':
                self.response_clients[name] = AsyncOpenaiResponseClient(base_url, api_key, **extract_args)
            else:
                self.logger.error(f"Unsupported response client type '{client['type']}'")
                exit(0)

            self.logger.info(f"Loaded response client : {name}")

    def _load_reason_clients(self, config_object):
        self.logger.info("Loading reason clients")
        reason_clients: List[Dict] = config_object.get("reason_clients", [])
        for client in reason_clients:
            name = client["name"]
            base_url = client["base_url"]
            api_key = client.get("api_key", "")
            extract_args = client.get("extract_args", {})


            if name in self.response_clients:
                self.logger.error(f"Detected duplicate reason clients : {name}")
                exit(0)

            if client["type"] == 'deepseek':
                self.reason_clients[name] = AsyncDeepseekReasonClient(base_url, api_key, **extract_args)
            elif client["type"] == 'openai':
                self.reason_clients[name] = AsyncOpenaiReasonClient(base_url, api_key, **extract_args)
            else:
                self.logger.error(f"Unsupported reason client type '{client['type']}'")
                exit(0)

            self.logger.info(f"Loaded reason client : {name}")

    def add_reason_client(self,name:str,client:AsyncReasonClient):
        self.reason_clients[name] = client

    def add_response_client(self,name:str,client:AsyncResponseClient):
        self.response_clients[name] = client

    def add_model(self,name:str,model:ModelInfo):
        self.models[name] = model

    @staticmethod
    def _extract_token(authorization:str):
        if (authorization is None) or (not authorization.startswith("Bearer ")):
            return None
        return authorization[7:]
    def _verify_authorization(self, authorization:Optional[str]):
        token  = DeepAnythingServer._extract_token(authorization)

        if not self.api_keys:
            return DeepAnythingServer._extract_token(authorization)

        if authorization is None or token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Expect token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if token not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token

    async def chat_completions(
            self,
            request: Request,  # 新增加Request参数
            query: Types.ChatCompletionQuery,
            authorization: Optional[str] = Header(None)
    ):
        token = self._verify_authorization(authorization)

        self.logger.info(f"ChatCompletions : {token} -> {query.model}")

        if query.model not in self.models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model not found",
            )

        model = self.models[query.model]

        async def _sse_warp(it: AsyncStream, req: Request):
            async for chunk in it:
                if await req.is_disconnected():
                    await it.close()
                    break
                yield f"data: {chunk.model_dump_json(indent=None)}\n\n".encode("utf-8")
            yield "data: [DONE]".encode('utf-8')

        args = DeepAnythingServer._extract_args(query)

        max_tokens = None
        if "max_tokens" in args:
            max_tokens = args["max_tokens"]
            args.pop("max_tokens")

        if query.stream:
            res = _sse_warp(
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
                    max_tokens=max_tokens,

                ),
                request
            )
            return StreamingResponse(
                res,
                media_type="text/event-stream",
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
            return Response(
                content=res.model_dump_json(indent=None),
                media_type="application/json"
            )

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