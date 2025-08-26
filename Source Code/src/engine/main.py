import asyncio
import aio_pika
import os
import sys
from loguru import logger
from config.config_handler import config
from core.queue_utils import process_message

# CRITICAL: Import and register Emotic class in __main__ context
import torch
import torch.nn as nn

# Define Emotic class in __main__ module context
class Emotic(nn.Module):
    def __init__(self, num_context_features, num_body_features):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out

# Make it globally available
globals()['Emotic'] = Emotic
torch.serialization.add_safe_globals([Emotic])

# Setup logging (rest of your original code stays the same)
if os.environ.get("MODE", "dev") == "prod":
    log_dir = "/approot/data"
else:
    log_dir = "../../../Outputs/result"
os.makedirs(log_dir, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config.get("CONSOLE_LOG_LEVEL", "INFO"),
    backtrace=True,
    diagnose=True,
    colorize=True,
    serialize=False,
    enqueue=True,
)
logger.add(
    f"{log_dir}/engine.log",
    rotation="50MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config.get("FILE_LOG_LEVEL", "INFO"),
    backtrace=True,
    diagnose=False,
    colorize=True,
    serialize=False,
    enqueue=True,
)

async def main():
    """Main worker function that runs in a continuous loop"""
    connection = None
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
        logger.info("Connected to RabbitMQ")

        # Create channel
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        # Declare main queue
        queue = await channel.declare_queue(
            "videototxt_queue",
            durable=True
        )

        # Declare result queue
        result_channel = await connection.channel()
        await result_channel.declare_queue(
            "emotion_detection_result_queue",
            durable=True
        )

        logger.info("Worker started. Waiting for messages...")

        # Continuous message processing loop
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    await process_message(message, result_channel)
                except Exception as e:
                    logger.exception(f"Failed to process message: {e}")
                    await message.nack()

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker stopping due to interrupt...")
    except Exception as e:
        logger.exception(f"Worker failed: {e}")
    finally:
        try:
            if connection and not connection.is_closed:
                await connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped gracefully")
    finally:
        loop.close()