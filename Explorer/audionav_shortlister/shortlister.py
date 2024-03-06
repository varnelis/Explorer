import io
import os
import json
import numpy as np
from tqdm import tqdm
from Explorer.db.mongo_db import MongoDBInterface

from Explorer.audionav_shortlister.ui_models_distributed import *

from typing import Union, Literal
from uuid import UUID
from dacite import from_dict
from PIL import Image, ImageDraw
from torchvision import transforms

import seaborn as sns
import matplotlib.pylab as plt

def from_dict_list(_type, data: list) -> list:
    return [from_dict(_type, d) for d in data]

ShortlisterType = Literal["ocr", "interactable-detector", "web7kbal", "web350k", "vins"]
InteractableModel = Literal["interactable-detector", "web7kbal", "web350k", "vins"]


class Shortlister:
    def __init__(
        self,
        screenshots_path: Union[str | None] = None,
        ocr_path: Union[str | None] = None,
    ) -> None:
        super(Shortlister).__init__()

        self.transforms = transforms.ToTensor()
        self.shortlist_bbox = {
            "ocr": [],
            "interactable-detector": [],
            "web7kbal": [],
            "web350k": [],
            "vins": [],
        }

    def get_image_ocr_by_uuid(
        self,
        uuid: Union[str | UUID],
        show_image: bool,
        save_image: Union[str | None] = None,
    ) -> None:
        """Image & OCR by querying MongoDB database for UUID"""

        MongoDBInterface.connect()
        screenshot = MongoDBInterface.get_items({"uuid": uuid}, "screenshots").limit(1)
        screenshot = list(screenshot)[0]
        image = screenshot["image"]
        image_id = screenshot["_id"]  # used to get corresponding OCR

        self.image = Image.open(io.BytesIO(image))
        if show_image:
            self.image.show()
        if save_image:
            self.image.save(f"./shortlist_images/image_raw_{uuid}.png")

        ocr_data = MongoDBInterface.get_items(
            {"screenshot_id": image_id}, "image-text-content"
        )
        self.ocr_data = list(ocr_data)[0]

    def shortlist_ocr(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        """Shortlist based on all unranked OCR bbox above confidence threshold"""

        confidence = self.ocr_data["confidence"]
        bbox_location = self.ocr_data["text_location"]

        for i in range(len(confidence)):
            if confidence[i] > confidence_threshold:
                bbox = bbox_location[i]
                xmin = bbox[0][0]
                xmax = bbox[1][0]
                ymin = bbox[0][1]
                ymax = bbox[1][1]

                self.shortlist_bbox["ocr"].append((xmin, ymin, xmax, ymax))

        self._non_max_suppression("ocr", nms_iou_threshold)

    def shortlist_interactables(
        self,
        model: InteractableModel,
        interactable_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        """Shortlist based on an interactable detector model (ours or WebUI)
        above confidence interactable_threshold (plus NMS up to nms_iou_threshold)."""

        if model == "interactable-detector":
            model_ckpt = "./last-v8.ckpt"
        elif model == "web7kbal":
            model_ckpt = "./screenrecognition-web7kbal.ckpt"
        elif model == "web350k":
            model_ckpt = "./screenrecognition-web350k.ckpt"
        elif model == "vins":
            model_ckpt = "./screenrecognition-web350k-vins.ckpt"
        else:
            raise Exception(
                f"`model` takes 'interactable-detector', 'web7kbal', 'web350k' or 'vins', got {model}"
            )

        img_input = self.transforms(self.image.copy().convert("RGB"))

        m = UIElementDetector.load_from_checkpoint(model_ckpt).eval()
        output_bbox_pred = m.model([img_input])

        bbox_preds_all = output_bbox_pred[0]["boxes"].tolist()
        bbox_scores = output_bbox_pred[0]["scores"].detach().numpy()

        try:
            last_valid_idx = np.where(bbox_scores < interactable_threshold + 1e-6)[0][
                0
            ]  # get last idx for boxes with confidence >= 0.5
            self.shortlist_bbox[model] = bbox_preds_all[:last_valid_idx]
        except IndexError:  # all bbox preds higher than threshold
            self.shortlist_bbox[model] = bbox_preds_all

        self._non_max_suppression(model, nms_iou_threshold)

    def _non_max_suppression(self, model: ShortlisterType, iou_threshold: float = 0.2):
        """NMS for bbox predictions on image. Drops low-confidence bbox with IoU overlap > threshold"""

        keep_bbox = []
        # self.shortlist_bbox and bbox_confidences are already sorted
        while self.shortlist_bbox[model]:
            i = self.shortlist_bbox[model].pop(0)
            keep_bbox.append(i)
            removals = []
            for j in self.shortlist_bbox[model]:
                iou = self._iou(bbox1=i, bbox2=j)
                if iou > iou_threshold:
                    removals.append(j)
            for j in removals:
                self.shortlist_bbox[model].remove(j)

        self.shortlist_bbox[model] = keep_bbox

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

    def draw_bbox(
        self, model: ShortlisterType, show_image: bool, save_image: Union[str | None]
    ):
        if len(self.shortlist_bbox[model]) == 0:
            raise Exception("bbox passed is empty. Shortlisting not yet performed.")

        imgcpy = self.image.copy()  # copy as ImageDraw modifies inplace

        draw_bbox = ImageDraw.Draw(imgcpy)
        for i, bbox in enumerate(self.shortlist_bbox[model]):
            draw_bbox.rectangle(bbox, outline="red", width=3)

        if show_image:
            imgcpy.show()
        if save_image:
            imgcpy.save("./shortlist_images/" + save_image + ".png")

    def get_shortlist(
        self,
        uuid: str,
        shortlist_method: ShortlisterType,
        shortlist_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
        show_image: bool = False,
        save_image: bool = False,
    ):
        """Get shortlist of interactable bbox to show, based on shortlisting method"""
        if not os.path.exists("./shortlist_images/"):
            os.mkdir("./shortlist_images/")
        self.get_image_ocr_by_uuid(uuid, show_image, save_image)
        if shortlist_method == "ocr":
            self.shortlist_ocr(
                confidence_threshold=shortlist_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
        else:
            self.shortlist_interactables(
                model=shortlist_method,
                interactable_threshold=shortlist_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
        if show_image or save_image:
            if save_image:
                save_name = f"image_shortlist_{shortlist_method}_{uuid}"
            self.draw_bbox(shortlist_method, show_image, save_name)

        return self.shortlist_bbox[shortlist_method]


if __name__ == "__main__":
    shortlister = Shortlister()
    uuid = "0a09f4fba5ae430c97c0c1d8c74301c8"
    shortlister.get_shortlist(uuid, "ocr", save_image=True)
    shortlister.get_shortlist(uuid, "interactable-detector", save_image=True)
    shortlister.get_shortlist(uuid, "web7kbal", save_image=True, shortlist_threshold=0.1)
    shortlister.get_shortlist(uuid, "web350k", save_image=True, shortlist_threshold=0.1)
    shortlister.get_shortlist(uuid, "vins", save_image=True, shortlist_threshold=0.3)