import os
import warnings
import numpy as np
from PIL import Image, ImageDraw
from Explorer.trace_similarity.screen_similarity import ScreenSimilarity
from Explorer.overlay.shortlister import Shortlister
from typing import Literal

bbox_type = tuple[float, float, float, float]
mode_type = Literal['black_full', 'cropped_subimg', 'resized_full']

class ActionMatching(ScreenSimilarity):

    def __init__(self) -> None:
        super().__init__()

    def _check_point_in_bbox(self, point: tuple[float, float], bbox: bbox_type) -> bool:
        (point_x, point_y) = point
        if (
            point_x >= bbox[0] and point_x <= bbox[2] and 
            point_y >= bbox[1] and point_y <= bbox[3]
        ):
            return True
        return False

    def get_interactables_on_image(self, image: Image.Image) -> list[bbox_type]:
        """ Gets list of bboxes of all interactables on the image. """
        shortlist_model = Shortlister()
        shortlist_model.set_model("interactable-detector")
        shortlist_model.set_img(image).set_bboxes()
        return shortlist_model.bboxes

    def get_interactable_at_click(self, image: Image.Image, click_pos: tuple[float, float]) -> None:
        """ Gets interactable bbox that contains the given click_pos.
        Used for User 1 image. """
        bboxes = self.get_interactables_on_image(image)
        for bbox in bboxes:
            if self._check_point_in_bbox(click_pos, bbox) is True:
                return bbox # return first bbox containing click_pos (bboxes ordered by confidence)
        warnings.warn('No bbox in given image contains given click_pos', category=RuntimeWarning)
        return None # no bbox fits
    
    def interactable2input(
            self, 
            image: Image.Image, 
            bbox: bbox_type,
            mode: mode_type,
        ) -> Image.Image:
        """ Single interactable at given bbox in given image converted to 
        entire-resolution input image """
        if mode=='black_full':
            img = np.array(image.copy())
            mask = np.zeros_like(img, dtype=np.uint8)
            bbox = list(map(lambda x: int(x), bbox)) # int for input-float bbox corners
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            return Image.fromarray(mask)
        elif mode=='cropped_subimg':
            return image.crop(bbox)
        elif mode=='resized_full':
            original_size = image.size
            return image.crop(bbox).resize(original_size)
        else:
            raise ValueError('mode arg invalid for interactable2input func')

    def interactable_matching(
        self, 
        image_user1: Image.Image, 
        image_user2: Image.Image, 
        click_user1: tuple[float, float],
        mode: mode_type,
    ) -> bbox_type:
        """ Return bbox coordinates of best-match interactable on User 2 image
        given User 1 clicked interactable. """
        interactable_user1 = self.get_interactable_at_click(image_user1, click_user1)
        input_image1 = self.interactable2input(image_user1, interactable_user1, mode)

        all_interactables_user2 = self.get_interactables_on_image(image_user2)
        min_dist = np.infty
        best_interactable = None
        for interactable_user2 in all_interactables_user2:
            input_image2 = self.interactable2input(image_user2, interactable_user2, mode)
            distances = self.trace_self_similarity([input_image1, input_image2])
            if distances[0] < min_dist:
                min_dist = distances[0]
                best_interactable = interactable_user2
        return best_interactable
    
    def show(self, image: Image.Image, bbox: bbox_type, bbox_color: str, savedir: str | None = None) -> None:
        draw_image = image.copy()
        draw_bbox = ImageDraw.Draw(draw_image)
        draw_bbox.rectangle(bbox, outline=bbox_color, width=5)
        
        if savedir is not None:
            draw_image.save(savedir)
        else:
            draw_image.show()