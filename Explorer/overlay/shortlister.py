import io
import os
import json
import gdown
import numpy as np
from tqdm import tqdm

from typing import Union, Literal
from uuid import UUID
from dacite import from_dict
from PIL import Image, ImageDraw
from torchvision import transforms

from Explorer.db.mongo_db import MongoDBInterface
from Explorer.ocr.ocr import OCR
from Explorer.overlay.ui_models_distributed import UIElementDetector

import seaborn as sns
import matplotlib.pylab as plt

def from_dict_list(_type, data: list) -> list:
    return [from_dict(_type, d) for d in data]

ShortlisterType = Literal["ocr", "interactable-detector", "web7kbal", "web350k", "vins"]
InteractableModel = Literal["interactable-detector", "web7kbal", "web350k", "vins"]

model2file = {
    "web7kbal": "screenrecognition-web7kbal.ckpt",
    "web350k": "screenrecognition-web350k.ckpt",
    "vins": "screenrecognition-web350k-vins.ckpt",
    "interactable-detector": "screenrecognition-interactdetect.ckpt",
}

model2version = {
    "web7kbal": "v0.1.0",
    "web350k": "v0.2.0",
    "vins": "v0.3.0",
    "interactable-detector": "v0.4.0",
}

class Shortlister:
    def __init__(self) -> None:
        if not os.path.exists("./shortlist_images/"):
            os.makedirs("./shortlist_images/")
        
        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        self.to_tensor = transforms.ToTensor()

        self.bboxes: list | None = None
        self.img_w_bboxes: Image.Image | None = None
        self.model: ShortlisterType | None = None
        self.img: Image.Image | None = None

    def set_model(self, model: ShortlisterType) -> "Shortlister":
        self.model = model
        return self

    def set_img(self, img: Image.Image) -> "Shortlister":
        self.img = img
        return self

    def set_bboxes(self) -> "Shortlister":
        if self.model is None:
            raise IndexError("Model was not chosen, call Shortlister.set_model() first")
        if self.img is None:
            raise IndexError("Image was not loaded, call Shortlister.set_img() first")
        
        self.img_w_bboxes = None
        if self.model == "ocr":
            bboxes = self._get_bboxes_ocr()
        else:
            bboxes = self._get_bboxes_webui()
        self.bboxes = bboxes

        return self
    
    def show(self) -> "Shortlister":
        if self.img_w_bboxes is None:
            self._add_bboxes_to_img()
        
        self.img_w_bboxes.show()

    def save(self) -> "Shortlister":
        if self.img_w_bboxes is None:
            self._add_bboxes_to_img()
        
        self.img_w_bboxes.save(f"./shortlist_images/{self.model}.png")

    def _add_bboxes_to_img(self):
        if self.bboxes is None:
            raise IndexError("Bounding boxes were not genreted, call Shortlister.set_bboxes() first")

        imgcpy = self.img.copy()  # copy as ImageDraw modifies inplace

        draw_bbox = ImageDraw.Draw(imgcpy)
        for bbox in self.bboxes:
            draw_bbox.rectangle(bbox, outline="red", width=3)

        self.img_w_bboxes = imgcpy

    def _get_bboxes_webui(
        self,
        interactable_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        """Shortlist based on an interactable detector model (ours or WebUI)
        above confidence interactable_threshold (plus NMS up to nms_iou_threshold)."""

        self._get_model_weights(self.model)
        m = UIElementDetector.load_from_checkpoint(self.model_weights_path).eval()
        
        img_input = self.to_tensor(self.img.copy().convert("RGB"))
        output_bbox_pred = m.model([img_input])
        bbox_preds_all = output_bbox_pred[0]["boxes"].tolist()
        bbox_scores = output_bbox_pred[0]["scores"].detach().numpy()

        try:
            # get last idx for boxes with confidence >= 0.5
            last_valid_idx = np.where(bbox_scores < interactable_threshold + 1e-6)[0][0]
            bboxes = bbox_preds_all[:last_valid_idx]
        except IndexError:  # all bbox preds higher than threshold
            bboxes = bbox_preds_all

        return self._non_max_suppression(bboxes, nms_iou_threshold)
    
    def _get_bboxes_ocr(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        """Shortlist based on all unranked OCR bbox above confidence threshold"""

        _, ocr_bboxes, confidence = OCR.get_text(self.img)
        bboxes = []

        for c, bbox in zip(confidence, ocr_bboxes):
            if c < confidence_threshold:
                continue
            bboxes.append(bbox)

        return self._non_max_suppression(bboxes, nms_iou_threshold)

    """def get_image_ocr_by_uuid(
        self,
        uuid: Union[str | UUID],
        show_image: bool,
        save_image: Union[str | None] = None,
    ) -> None:
        "Image & OCR by querying MongoDB database for UUID"

        MongoDBInterface.connect()
        screenshot = MongoDBInterface.get_items({"uuid": uuid}, "screenshots").limit(1)
        screenshot = list(screenshot)[0]
        image = screenshot["image"]
        image_id = screenshot["_id"]  # used to get corresponding OCR

        self.image = Image.open(io.BytesIO(image))
        if show_image:
            self.image.show()
        if save_image:
            self.image.save(os.path.join(self.base_dir, f"./shortlist_images/image_raw_{uuid}.png"))

        ocr_data = MongoDBInterface.get_items(
            {"screenshot_id": image_id}, "image-text-content"
        )
        self.ocr_data = list(ocr_data)[0]"""
    
    def _get_model_weights(self, model: InteractableModel) -> None:
        """ Get file for weights of the model (local or from mongodb) """
        
        MongoDBInterface.connect()
        weights = MongoDBInterface.get_items({"version": model2version[model]}, "detectors").limit(1)
        weights = list(weights)[0]

        self.model_weights_path = self._gdown_url(weights["url"], model2file[model])

    def _gdown_url(self, url: str, model_key: str) -> None:
        if not os.path.exists(os.path.join("./weights", model_key)):
            #url = url + '/view?usp=share_link'
            print('Retrieving weights from URL = ', url)
            gdown.download(url=url, output=os.path.join("./weights", model_key), fuzzy=True)
        return os.path.join("./weights", model_key)

    def _non_max_suppression(self, bboxes, iou_threshold: float = 0.2):
        """NMS for bbox predictions on image. Drops low-confidence bbox with IoU overlap > threshold"""

        keep_bbox = []
        # self.shortlist_bbox and bbox_confidences are already sorted
        while len(bboxes) != 0:
            bbox_1 = bboxes.pop(0)
            keep_bbox.append(bbox_1)
            removals = []
            for bbox_2 in bboxes:
                iou = self._iou(bbox_1, bbox_2)
                if iou > iou_threshold:
                    removals.append(bbox_2)
            for bbox in removals:
                bboxes.remove(bbox)

        return keep_bbox

    def _iou(self, bbox1, bbox2):
        """Get IoU of 2 bbox of format (xmin, ymin, xmax, ymax)"""

        # intersection
        intersect_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
        intersect_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
        if intersect_height <= 0 or intersect_width <= 0:
            return 0
        intersection = intersect_width * intersect_height

        # union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union
