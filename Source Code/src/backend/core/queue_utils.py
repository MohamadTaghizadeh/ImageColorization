import aio_pika
import json
from loguru import logger
from typing import AsyncGenerator
from config.config_handler import config
from core import webhook_handler
from sqlalchemy.orm import Session
from dbutils import crud
from dbutils.schemas import WebhookStatus


async def consume_results(connection: aio_pika.RobustConnection, db: Session):
    """Consume results from the result queue and update the database."""
    async with connection:
        channel = await connection.channel()
        result_queue = await channel.declare_queue("emotion_detection_result_queue", durable=True)

        async for message in result_queue:
            async with message.process():
                try:
                    result = json.loads(message.body.decode())
                    request_id = result["request_id"]
                    logger.info(f"Processing result for {request_id}")

                    # Process status and update database
                    status_map = {
                        "pending": WebhookStatus.pending,
                        "in_progress": WebhookStatus.in_progress,
                        "completed": WebhookStatus.completed,
                        "failed": WebhookStatus.failed
                    }
                    status = status_map.get(result["status"], WebhookStatus.failed)

                    crud.update_request(
                        db=db,
                        request_id=request_id,
                        status=status,
                        result=result.get("results"),
                        error=result.get("error"),
                    )

                    # Handle webhook status
                    if status == WebhookStatus.completed:
                        webhook_handler.set_completed(request_id=request_id, db=db)
                    elif status == WebhookStatus.failed:
                        webhook_handler.set_failed(request_id=request_id)

                except Exception as e:
                    logger.exception(f"Error processing result: {e}")


async def send_video_task(video_data: str, request_id: str, experiment_path: str):
    """Send video processing task to queue"""
    connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
    try:
        channel = await connection.channel()
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({
                    "request_id": request_id,
                    "video": video_data,
                    "experiment_path": experiment_path
                }).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key="emotion_detection_queue"
        )
        logger.info(f"Sent video task {request_id}")
    finally:
        await connection.close()


async def get_rabbitmq_connection() -> AsyncGenerator[aio_pika.RobustConnection, None]:
    """Dependency to get RabbitMQ connection"""
    connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
    try:
        yield connection
    finally:
        await connection.close()