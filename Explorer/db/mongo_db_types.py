

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from bson import ObjectId

@dataclass
class Scanner:
    version: str
    datetime: datetime
    _id: ObjectId | None = None

@dataclass
class ScannedPage:
    uuid: UUID
    url: str
    datetime: datetime
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class Platform:
    os: str
    os_version: str
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class OCRModel:
    version: str
    datetime: datetime
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class Screenshot:
    uuid: UUID
    platform_id: ObjectId
    image_hash: str
    image: bytes
    _id: ObjectId | None = None

@dataclass
class ScannerBoundingBoxes:
    scanner_version: str
    screenshot_id: ObjectId
    bboxes: list[tuple[int, int, int, int]]
    image_hash: str
    image: bytes
    _id: ObjectId | None = None

@dataclass
class ImageTextContent:
    screenshot_id: ObjectId
    ocr_version: str
    text: list[str]
    text_location: list
    confidence: list[float]
    _id: ObjectId | None = None

@dataclass
class DetectorTrainingSample:
    screenshot_id: ObjectId
    detector_version: str
    sample_type: str
    _id: ObjectId | None = None

@dataclass
class DetectorModel:
    version: str
    datetime: datetime
    url: str
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class ScreenCaptureModel:
    version: str
    datetime: datetime
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class InputCaptureModel:
    version: str
    datetime: datetime
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class Trace:
    platform_id: ObjectId
    screen_capture_version: str
    input_capture_version: str
    datetime: datetime
    metadata: dict | None = None
    _id: ObjectId | None = None

@dataclass
class TraceSnapshot:
    trace_id: ObjectId
    image_hash: str
    image: bytes
    timestamp: int
    _id: ObjectId | None = None

@dataclass
class InputT:
    timestamp: int
    x: float
    y: float
    mouse_state: int
    spk_state: int
    kv_state: int

@dataclass
class TraceInputs:
    trace_id: ObjectId
    input: InputT
    _id: ObjectId | None = None

