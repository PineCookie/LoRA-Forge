import asyncio
import mimetypes
import os
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from lora_forge.app.api import router as api_router
from lora_forge.app.proxy import close_proxy_clients, router as proxy_router
from lora_forge.devices import check_torch_gpu
from lora_forge.frontend import DIST_DIR
from lora_forge.i18n import LocaleMiddleware, t

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

FRONTEND_MISSING_PAGE = """<!doctype html>
<meta charset="utf-8"><title>LoRA-Forge</title>
<body style="font-family: system-ui, sans-serif; max-width: 640px; margin: 64px auto; line-height: 1.6">
<h1>{title}</h1>
<p>{body}</p>
</body>"""


async def app_startup():
    await asyncio.to_thread(check_torch_gpu)

    if os.environ.get("LORA_FORGE_OPEN_BROWSER", "0") == "1":
        webbrowser.open(f'http://{os.environ["LORA_FORGE_HOST"]}:{os.environ["LORA_FORGE_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    try:
        yield
    finally:
        await close_proxy_clients()


# The UI is served from this app (or proxied by Vite's dev server), so no CORS is needed.
app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    # built assets have content hashes in their names; everything else must be revalidated
    if request.url.path.startswith("/assets/") and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response


# added last, so it is the outermost middleware: every response is built in the request's language
app.add_middleware(LocaleMiddleware)
app.include_router(api_router, prefix="/api")


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
async def favicon():
    return FileResponse("assets/favicon.ico")


app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets", check_dir=False), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str):
    # unknown API / proxy paths are real 404s, not the app shell
    if full_path.startswith(("api/", "proxy/")):
        raise HTTPException(status_code=404)
    index = DIST_DIR / "index.html"
    if not index.is_file():
        page = FRONTEND_MISSING_PAGE.format(title=t("page.frontendMissing.title"), body=t("page.frontendMissing.body"))
        return HTMLResponse(page, status_code=503)
    # every page path is a client-side route of the single-page app
    return FileResponse(index)
