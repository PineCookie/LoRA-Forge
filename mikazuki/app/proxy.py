import os
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, Request
from httpx import HTTPError
from starlette.background import BackgroundTask
from starlette.responses import PlainTextResponse, StreamingResponse

from mikazuki.log import log

router = APIRouter()

PROXY_TIMEOUT_SECONDS = 360
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
HOP_BY_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


@dataclass(frozen=True)
class ProxyTarget:
    host_env: str
    port_env: str
    default_host: str
    default_port: str


PROXY_TARGETS = {
    "tensorboard": ProxyTarget(
        host_env="MIKAZUKI_TENSORBOARD_HOST",
        port_env="MIKAZUKI_TENSORBOARD_PORT",
        default_host="127.0.0.1",
        default_port="6006",
    ),
}
_proxy_clients: list[httpx.AsyncClient] = []


def _client_for_target(url_type: str) -> httpx.AsyncClient:
    target = PROXY_TARGETS.get(url_type)
    if target is None:
        raise ValueError(f"Unknown proxy target: {url_type}")

    host = os.environ.get(target.host_env, target.default_host)
    port = os.environ.get(target.port_env, target.default_port)
    client = httpx.AsyncClient(
        base_url=f"http://{host}:{port}/",
        trust_env=False,
        timeout=PROXY_TIMEOUT_SECONDS,
    )
    _proxy_clients.append(client)
    return client


async def close_proxy_clients():
    for client in _proxy_clients:
        await client.aclose()
    _proxy_clients.clear()


def _proxy_path(request: Request, full_path: bool) -> str:
    if full_path:
        return request.url.path

    path = request.path_params.get("path", "")
    return "/" if path == "" else f"/{path.lstrip('/')}"


def _filtered_headers(raw_headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    return [
        (name, value)
        for name, value in raw_headers
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != b"host"
    ]


def reverse_proxy_maker(url_type: str, full_path: bool = False):
    client = _client_for_target(url_type)

    async def _reverse_proxy(request: Request):
        url = httpx.URL(
            path=_proxy_path(request, full_path),
            query=request.url.query.encode("utf-8"),
        )
        rp_req = client.build_request(
            request.method,
            url,
            headers=_filtered_headers(request.headers.raw),
            content=None if request.method in {"GET", "HEAD"} else request.stream(),
        )
        try:
            rp_resp = await client.send(rp_req, stream=True)
        except HTTPError as e:
            log.warning(f"Reverse proxy failed for {url_type}: {e}")
            return PlainTextResponse(
                content="The requested service not started yet or service started fail. This may cost a while when you first time startup\n请求的服务尚未启动或启动失败。若是第一次启动，可能需要等待一段时间后再刷新网页。",
                status_code=502
            )
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=_filtered_headers(rp_resp.headers.raw),
            background=BackgroundTask(rp_resp.aclose),
        )

    return _reverse_proxy


router.add_route("/proxy/tensorboard/{path:path}", reverse_proxy_maker("tensorboard"), PROXY_METHODS)
router.add_route("/font-roboto/{path:path}", reverse_proxy_maker("tensorboard", full_path=True), PROXY_METHODS)
