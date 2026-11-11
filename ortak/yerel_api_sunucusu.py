"""Local HTTP endpoints for model inference."""
import base64
import binascii
import re
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from ortak.calisma_ortami import (
    ENGINES,
    RESTORERS,
    DemoError,
    error_response,
    quiet_libraries,
)
from ortak.gorsel_okuma import MAX_BYTES
from ortak.motor_servisi import InferenceService

MAX_REQUEST = 24 * 1024 * 1024
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
TargetText = Annotated[str, Field(min_length=1, max_length=100)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    engine: Literal['grounding-dino-tiny', 'ppocr-v6-medium']
    image_base64: str = Field(min_length=1, max_length=((MAX_BYTES+2)//3)*4)
    prompt: str = Field(default='', max_length=250)
    targets: list[TargetText] = Field(default_factory=list, max_length=20)
    threshold: Score = .5
    text_threshold: Score = .25
    nms: Score = .5


class RestorationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    engine: Literal['nafnet-reds','concertormer-reds']
    image_base64: str = Field(min_length=1,max_length=((MAX_BYTES+2)//3)*4)
    blend: Annotated[float,Field(ge=.1,le=1,allow_inf_nan=False)] = 1.


class RequestBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope['headers']}
        host = headers.get(b'host', b'').decode('latin-1')
        origin = headers.get(b'origin', b'').decode('latin-1')
        if not re.fullmatch(r'(127\.0\.0\.1|localhost)(:[0-9]{1,5})?', host) or (origin and origin not in {f'http://{host}'}):
            return await JSONResponse(error_response('invalid_request'), status_code=403)(scope, receive, send)
        if scope['method'] == 'POST':
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > MAX_REQUEST:
                    return await JSONResponse(error_response('input_too_large'), status_code=413)(scope, receive, send)
                if not message.get('more_body', False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if delivered:
                    return {'type': 'http.disconnect'}
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def create_app(model_root, service=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, debug=False)
    app.add_middleware(RequestBoundary)
    inference = service or InferenceService(model_root)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Use the common error response format.
        return JSONResponse(error_response('invalid_request'), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(error_response('invalid_request'), status_code=error.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse(error_response('operation_failed'), status_code=500)

    @app.get('/health')
    async def health():
        return {'ok': True, 'engines': list(ENGINES+RESTORERS)}

    @app.post('/predict')
    async def predict(payload: PredictionRequest):
        try:
            try:
                data = base64.b64decode(payload.image_base64, validate=True)
            except (ValueError, binascii.Error):
                raise DemoError('invalid_image') from None
            result = await run_in_threadpool(inference.predict, payload.engine, data, prompt=payload.prompt,
                targets=payload.targets, threshold=payload.threshold, text_threshold=payload.text_threshold, nms=payload.nms)
            return JSONResponse(result, headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        except DemoError as error:
            status = 503 if error.code in ('model_missing', 'model_integrity', 'dependencies_missing') else 400
            return JSONResponse(error_response(error.code), status_code=status)
        except ImportError:
            return JSONResponse(error_response('dependencies_missing'), status_code=503)
        except Exception:
            return JSONResponse(error_response('inference_failed'), status_code=500)

    @app.post('/restore')
    async def restore(payload: RestorationRequest):
        try:
            try:
                data = base64.b64decode(payload.image_base64,validate=True)
            except (ValueError,binascii.Error):
                raise DemoError('invalid_image') from None
            result = await run_in_threadpool(inference.restore,payload.engine,data,blend=payload.blend)
            return JSONResponse(result,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
        except DemoError as error:
            status = 503 if error.code in ('model_missing','model_integrity','dependencies_missing') else 400
            return JSONResponse(error_response(error.code),status_code=status)
        except ImportError:
            return JSONResponse(error_response('dependencies_missing'),status_code=503)
        except Exception:
            return JSONResponse(error_response('inference_failed'),status_code=500)

    return app


def serve(model_root, port):
    import uvicorn
    app = create_app(model_root)
    # The model runners are reused. Process-wide library output is discarded.
    with quiet_libraries():
        uvicorn.run(app, host='127.0.0.1', port=port, access_log=False, log_config=None,
                    log_level='critical', server_header=False, date_header=False,
                    proxy_headers=False, limit_concurrency=4)
