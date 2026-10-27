"""Flower prediction and grounded care API. An LLM is optional."""
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import logging
from pathlib import Path
import warnings

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.inference import Predictor
from app.care import Flower, build_care_graph

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000
logger = logging.getLogger(__name__)


class CareRequest(BaseModel):
    flower: Flower | None = None
    question: str = Field(default="", max_length=400)


def decode_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "Image exceeds 12 million pixels.")
                image.load()
                ImageOps.exif_transpose(image, in_place=True)
                return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Upload a valid JPEG or PNG image.") from None
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "Image dimensions are too large.") from None


def create_app(predictor=None):
    predictor = predictor if predictor is not None else Predictor()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(predictor.load)
        yield

    app = FastAPI(title="Flower Lab API", version="0.3.0", lifespan=lifespan)
    app.state.predictor = predictor
    care_graph = build_care_graph()
    prediction_slots = asyncio.Semaphore(1)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(static_dir / "index.html")

    @app.post("/care/chat")
    def care_chat(request: CareRequest):
        return care_graph.invoke(request.model_dump(), {"recursion_limit": 8})["response"]

    @app.get("/health")
    def health():
        result = {"status": "ok", "model_ready": predictor.ready}
        checkpoint_sha = getattr(predictor, "checkpoint_sha256", None)
        if predictor.ready and checkpoint_sha:
            result["checkpoint_sha256"] = checkpoint_sha
        return result

    @app.get("/ready")
    def ready():
        if not predictor.ready:
            raise HTTPException(503, predictor.error)
        return {"status": "ready"}

    @app.get("/classes")
    def classes():
        if not predictor.ready:
            raise HTTPException(503, predictor.error)
        return {"classes": predictor.classes}

    @app.post("/predict")
    async def predict(file: UploadFile):
        # Keep image decoding + inference together so large decoded buffers do
        # not accumulate while other requests wait for the model lock.
        async with prediction_slots:
            try:
                data = await file.read(MAX_BYTES + 1)
            finally:
                await file.close()
            if not data:
                raise HTTPException(400, "The uploaded file is empty.")
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "Upload must be 5 MiB or smaller.")
            image = await run_in_threadpool(decode_image, data)
            del data
            try:
                if not predictor.ready:
                    raise HTTPException(503, predictor.error)
                try:
                    return await run_in_threadpool(predictor.predict, image)
                except Exception:
                    logger.exception("Prediction failed")
                    raise HTTPException(500, "Prediction failed. Check server logs.") from None
            finally:
                image.close()

    return app


app = create_app()
