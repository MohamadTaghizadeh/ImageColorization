from generator import EmotionDetector
import aio_pika
import json
import base64
import os
from loguru import logger
from datetime import datetime
import tempfile

if os.environ.get("MODE", "dev") == "prod":
    output_dir = "/approot/data/result"
else:
    output_dir = "../../../Outputs/result"
os.makedirs(output_dir, exist_ok=True)

async def process_message(message: aio_pika.IncomingMessage, result_channel: aio_pika.Channel):
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            request_id = body["request_id"]
            encoded_video = body["video"]
            experiment_path = body.get("experiment_path", "default")  # Use .get() for safety
            
            logger.info(f"Processing request {request_id}")

            # Send processing status
            await result_channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "request_id": request_id,
                        "status": "in_progress"
                    }).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key="emotion_detection_result_queue"
            )

            # Process video
            results = await process_video_task(encoded_video, experiment_path)
            
            logger.info(f"Video processing results for {request_id}: {results}")

            # Prepare and send result
            if "error" in results:
                result_data = {
                    "request_id": request_id,
                    "status": "failed",
                    "error": results["error"]
                }
            else:
                result_data = {
                    "request_id": request_id,
                    "status": "completed",
                    "results": results  # Note: "results" not "result"
                }

            await result_channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(result_data).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key="emotion_detection_result_queue"
            )
            
            logger.info(f"Sent result for request {request_id}")

        except Exception as e:
            logger.exception(f"Failed to process message: {e}")
            try:
                # Send error result
                await result_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps({
                            "request_id": request_id if 'request_id' in locals() else "unknown",
                            "status": "failed",
                            "error": str(e)
                        }).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                    ),
                    routing_key="emotion_detection_result_queue"
                )
            except:
                logger.error("Failed to send error result")


async def process_video_task(encoded_video, experiment_path):
    """Process video and return emotion analysis results"""
    try:
        logger.info("Starting video processing task")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_data = base64.b64decode(encoded_video)
            temp_file.write(video_data)
            temp_path = temp_file.name

        logger.info(f"Created temporary video file: {temp_path}")
        
        detector = EmotionDetector(gpu=0)
        results = detector.process_video(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        logger.info("Video processing completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        return {"error": str(e)}