from pydantic import BaseModel
from typing import Optional, Dict


class APIResponse(BaseModel):
    status: str
    message: Optional[str]
    data: Optional[Dict]


class APIResponseSuccess(APIResponse):
    status: str = "success"
    message: Optional[str] = None
    data: Optional[Dict] = None


class APIResponseFail(APIResponse):
    status: str = "fail"
    message: Optional[str] = None
    data: Optional[Dict] = None

class APIResponsePending(APIResponse):
    status: str = "pending"
    message: Optional[str] = None
    data: Optional[Dict] = None