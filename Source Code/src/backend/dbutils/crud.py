from sqlalchemy.orm import Session
from dbutils import models
from loguru import logger
from core.messages import Message
from datetime import datetime, timedelta
from dbutils.schemas import WebhookStatus
from typing import Dict
from sqlalchemy.exc import IntegrityError
from core import utils


def clear_database(db: Session):
    try:
        db.query(models.Manager).delete()
        db.commit()
        return True
    except Exception as exp:
        logger.opt(exception=False, colors=True).warning(f"Failed: {exp.args}")
        db.rollback()
        return False


def get_request(
    db: Session,
    request_id: str,
):
    item = (
        db.query(models.Manager).filter(models.Manager.request_id == request_id).first()
    )
    return item


def add_request(db: Session, **kwargs):
    kwargs["itime"] = datetime.now(tz=None)
    item = models.Manager(**kwargs)
    try:
        db.add(item)
        db.commit()
        return Message("en").INF_SUCCESS()
    except IntegrityError as e:
        if "UNIQUE constraint failed: videototxt_manager.request_id" in str(e.args):
            msg = Message("en").ERR_DUPLICATE_REQUEST_ID()
            return msg
        else:
            logger.opt(exception=True).error("Failed to add_request")
            msg = Message("en").ERR_FAILED_TO_ADD_TO_DB()
            return msg
    except Exception:
        logger.opt(exception=True).error("Failed to add_request")
        msg = Message("en").ERR_FAILED_TO_ADD_TO_DB()
        return msg


def update_request(
    db: Session, request_id: str, status: WebhookStatus, result: Dict, error: str = None
):
    item = (
        db.query(models.Manager).filter(models.Manager.request_id == request_id).first()
    )
    if item is None:
        return False
    item.utime = datetime.now(tz=None)
    item.status = status
    item.result = result
    if error is not None:
        item.error = error
    try:
        db.commit()
        return True
    except Exception:
        logger.opt(exception=True, colors=True).error("Failed to update_request")
        return False


def set_webhook_result(db: Session, request_id: str, webhook_status_code: int) -> bool:
    item = (
        db.query(models.Manager).filter(models.Manager.request_id == request_id).first()
    )
    if item is None:
        return False
    item.utime = datetime.now(tz=None)
    if item.webhook_retry_count is None:
        item.webhook_retry_count = 0
    else:
        item.webhook_retry_count += 1
    item.webhook_status_code = webhook_status_code
    try:
        db.commit()
        return True
    except Exception:
        logger.opt(exception=True, colors=True).error("Failed to set_webhook_result")
        return False


def clean_unused_temp_files(db: Session):
    filepaths = (
        db.query(models.Manager.input_path)
        .filter(models.Manager.status == "completed")
        .filter(models.Manager.webhook_status_code == 200)
        .filter(models.Manager.utime < datetime.now() - timedelta(days=7))
        .all()
    )
    for filepath in filepaths:
        utils.delete_file(filepath[0])
