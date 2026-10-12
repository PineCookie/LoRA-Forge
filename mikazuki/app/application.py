import asyncio
import mimetypes
import os
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
from mikazuki.app.proxy import close_proxy_clients, router as proxy_router
from mikazuki.frontend import DIST_DIR
from mikazuki.utils.devices import check_torch_gpu

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

FRONTEND_MISSING_PAGE = """<!doctype html>
<meta charset="utf-8"><title>LoRA-Forge</title>
<body style="font-family: system-ui, sans-serif; max-width: 640px; margin: 64px auto; line-height: 1.6">
<h1>The web UI is not built</h1>
<p><code>frontend/dist</code> is missing. Either install <a href="https://nodejs.org/">Node.js</a> 20 or newer and
restart the GUI (it builds the UI automatically), or build it yourself with
<code>npm ci &amp;&amp; npm run build</code> in the <code>frontend</code> folder,
or use the ready-to-run zip from the GitHub Releases page.</p>
<p>网页界面尚未构建：请安装 Node.js 20+ 后重新启动 GUI（会自动构建），或在 <code>frontend</code> 目录运行
<code>npm ci &amp;&amp; npm run build</code>，或下载 Releases 中的完整压缩包。</p>
</body>"""


async def app_startup():
    app_config.load_config()

    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)

    if os.environ.get("MIKAZUKI_OPEN_BROWSER", "0") == "1":
        webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    try:
        yield
    finally:
        await close_proxy_clients()


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    # built assets have content hashes in their names; everything else must be revalidated
    if request.url.path.startswith("/assets/") and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response

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
        return HTMLResponse(FRONTEND_MISSING_PAGE, status_code=503)
    # every page path is a client-side route of the single-page app
    return FileResponse(index)
