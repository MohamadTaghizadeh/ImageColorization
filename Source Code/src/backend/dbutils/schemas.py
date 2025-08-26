from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class WebhookStatus(Enum):
    """Status codes for the webhook"""
    pending = 0
    in_progress = 1
    completed = 2
    failed = 3


class VideoToTxtResult(BaseModel):
    """Schema for video to text results"""
    #pose_actions: List[str]
    #fall_status: bool
    #body_detected: bool
    #I have a JSON Output (modify above three line codes)


class VideoToTxtRequest(BaseModel):
    """Schema for video to text request"""
    request_id: Optional[str] = None
    priority: int = 1
    language: Optional[str] = "en"


class TaskStatus(BaseModel):
    """Schema for task status"""
    request_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    itime: datetime
    utime: Optional[datetime] = None
    results: Optional[VideoToTxtResult] = None
    error: Optional[str] = None


class WebhookRequest(BaseModel):
    """Schema for webhook request"""
    request_id: str


class WebhookResponse(BaseModel):
    """Schema for webhook response"""
    status: bool = False
    message: str = ""
    code: str = ""
    results: Optional[VideoToTxtResult] = None


class ApiResponse(BaseModel):
    """Standard API response format"""
    status: bool = False
    message: str = ""
    code: str = ""
    data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
