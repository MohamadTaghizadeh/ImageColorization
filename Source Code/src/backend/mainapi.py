import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from sqlalchemy.orm import Session
import aio_pika
import uuid
import json
import asyncio
import base64
from core.messages import Message
from datetime import datetime
from fastapi.responses import FileResponse
from loguru import logger
import os
from version import __version__
from config.config_handler import config
from core.queue_utils import consume_results, get_rabbitmq_connection
from dbutils import schemas, crud
import sys
from starlette.middleware.cors import CORSMiddleware
from core import base
from dbutils.database import SessionLocal


if os.environ.get("MODE", "dev") == "prod":
    log_dir = "/approot/data"
else:
    log_dir = "../../../Outputs/result"
os.makedirs(log_dir, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config["CONSOLE_LOG_LEVEL"],
    backtrace=True,
    diagnose=True,
    colorize=True,
    serialize=False,
    enqueue=True,
)
logger.add(
    f"{log_dir}/backend.log",
    rotation="50MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config["FILE_LOG_LEVEL"],
    backtrace=True,
    diagnose=False,
    colorize=False,
    serialize=False,
    enqueue=True,
)

logger.info("Starting Video to Text Service", version=__version__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan to start the result consumer."""
    try:
        connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
        db = SessionLocal()
        asyncio.create_task(consume_results(connection, db))
        yield
    finally:
        db.close()
        await connection.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
base_mdl = base.Base()


@app.post("/aihive-hrvidtxt/api/v1/video-to-txt-offline")
async def process_video(
        video: UploadFile = File(...),
        request_id: str = Form(None),
        priority: int = Form(1),
        connection: aio_pika.RobustConnection = Depends(get_rabbitmq_connection),
        db: Session = Depends(base.get_db),
):
    logger.info("/videototxt/video-to-txt-offline", video=video.filename, request_id=request_id)

    # Generate request_id if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())

    try:
        # Read the video
        video_data = await video.read()
    except Exception as e:
        logger.error(f"Failed to read video: {e}")
        return Message("fa").ERR_INVALID_INPUT()

    # Add request to database
    response = crud.add_request(
        db=db,
        request_id=request_id,
        status=schemas.WebhookStatus.pending,
        itime=datetime.now(tz=None),
    )

    if not response["status"]:
        return response

    # Create channel
    channel = await connection.channel()

    # Convert video to base64
    base64_video = base64.b64encode(video_data).decode('utf-8')

    # Prepare message
    message_body = {
        "video": base64_video,
        "request_id": request_id,
        "experiment_path": "default",
        "priority": priority,
    }

    # Publish to queue
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(message_body).encode(),
            headers={"request_id": request_id},
        ),
        routing_key="videototxt_queue",
    )
    await channel.close()

    # Return success response
    msg = Message("fa").INF_SUCCESS()
    msg["data"] = {"request_id": request_id}
    return msg


@app.get("/aihive-videototxt/api/v1/status/{request_id}")
async def get_status(request_id: str, db: Session = Depends(base.get_db)):
    logger.info("/videototxt/status", request_id=request_id)
    task = crud.get_request(db=db, request_id=request_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    msg = Message("fa").INF_SUCCESS()
    msg["data"] = task
    return msg




if __name__ == "__main__":
    uvicorn.run(app="mainapi:app", host="0.0.0.0", port=8000, reload=False)
