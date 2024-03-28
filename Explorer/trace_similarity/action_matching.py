import os
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from Explorer.trace_similarity.screen_similarity import ScreenSimilarity
from Explorer.overlay.shortlister import Shortlister

from typing import Literal

bbox_type = tuple[float, float, float, float]
mode_type = Literal['black_full', 'resized_full']

class ActionMatching(ScreenSimilarity):

    def __init__(self) -> None:
        super().__init__()
        self.shortlist_model = Shortlister()
        self.shortlist_model.set_model("interactable-detector")

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
        return self.shortlist_model.set_img(image).set_bboxes().bboxes

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
            show: bool,
        ) -> Image.Image:
        """ Single interactable at given bbox in given image converted to 
        entire-resolution input image """
        original_size = image.size
        if mode=='black_full':
            img = np.array(image)
            mask = np.zeros_like(img, dtype=np.uint8)
            bbox = [int(x) for x in bbox] # int for input-float bbox corners
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            ret_img = Image.fromarray(mask)
        elif mode=='resized_full':
            ret_img = image.crop(bbox)
        else:
            raise ValueError('mode arg invalid for interactable2input func')
        if show is True:
            ret_img.show()
        return ret_img.resize(original_size), ret_img
    
    def match_interactables_ocrs(self, interactables: list[bbox_type], ocrs: list[tuple[bbox_type, str]]) -> list[str]:
        """ Return array of str OCRs (concat), where OCR at index i corresponds to interactable at index i in interactables list. """
        text_for_interactable = [''] * len(interactables)
        for i, i_bbox in enumerate(interactables):
            for ocr in ocrs:
                ocr_bbox = ocr[0]
                ioa = Shortlister()._iou(i_bbox, ocr_bbox, ret_I_over_area2=True) # gives intersection over area of OCR bbox
                if ioa >= 0.5:
                    if text_for_interactable[i] == '':
                        text_for_interactable[i] += ocr[1]
                    else:
                        text_for_interactable[i] += ' ' + ocr[1]
        return text_for_interactable

    def interactable_matching(
        self, 
        image_user1: Image.Image, 
        image_user2: Image.Image, 
        click_user1: tuple[float, float],
        mode: mode_type,
        include_ocr: bool = 0,
        show_user1: bool = False,
        ret_top_k: int = 0,
    ) -> bbox_type:
        """ Return bbox coordinates of best-match interactable on User 2 image
        given User 1 clicked interactable. """
        interactable_user1 = self.get_interactable_at_click(image_user1, click_user1)
        input_image1, input_image1_originalsize = self.interactable2input(image_user1, interactable_user1, mode, show=show_user1)
        all_interactables_user2 = self.get_interactables_on_image(image_user2) # also do NMS?

        if include_ocr is True:
            ocr_bbox_text_image1 = self.image2ocr(input_image1_originalsize)
            interactable2ocr_image1 = [el[1] for el in ocr_bbox_text_image1]
            ocr_embedding_image1 = self.ocr2embedding(interactable2ocr_image1, concat=True)
            ocr_bbox_text_image2 = self.image2ocr(image_user2)
            interactable2ocr_image2 = self.match_interactables_ocrs(all_interactables_user2, ocr_bbox_text_image2)
            ocr_embeddings_image2 = self.ocr2embedding(interactable2ocr_image2, concat=False)
        
        dists = []
        min_dist = np.infty
        best_interactable = None
        
        for i in tqdm(range(len(all_interactables_user2)), desc='Comparing appearance'):
            interactable_user2 = all_interactables_user2[i]
            input_image2, _ = self.interactable2input(image_user2, interactable_user2, mode, show=False)
            distances = self.trace_self_similarity([input_image1, input_image2])
            total_dist = distances[0]

            if include_ocr is True:
                ocr_dist = self.encoding_distance(ocr_embedding_image1, ocr_embeddings_image2[i])
                print('ocr dist = ', ocr_dist)
                total_dist += ocr_dist
            
            if total_dist < min_dist:
                min_dist = total_dist
                best_interactable = interactable_user2

            dists.append(total_dist)

        if ret_top_k > 0:
            topk_idx = sorted(range(len(dists)), key=lambda i: dists[i])[:ret_top_k]
            topk_interactables = [all_interactables_user2[i] for i in topk_idx]
            return best_interactable, topk_interactables
        return best_interactable, None
    
    def show(
        self, 
        image: Image.Image, 
        bbox: bbox_type | list[bbox_type], 
        bbox_color: str | list[str], 
        savedir: str | None = None,
    ) -> None:
        draw_image = image.copy()
        draw_bbox = ImageDraw.Draw(draw_image)
        if all(isinstance(element, list) for element in bbox):
            if isinstance(bbox_color, str):
                bbox_color = [bbox_color] * len(bbox)
            assert(len(bbox)==len(bbox_color)), "Wrong dimensionality for color arg"
            for bb in range(len(bbox)):
                draw_bbox.rectangle(bbox[bb], outline=bbox_color[bb], width=5)
        else:
            draw_bbox.rectangle(bbox, outline=bbox_color, width=5)
        
        if savedir is not None:
            draw_image.save(savedir)
        else:
            draw_image.show()

    def replicate_action_on_given_state(
        self, 
        image1: Image.Image, 
        image2: Image.Image, 
        action_on_image1: tuple[float, float],
        mode: mode_type,
        include_ocr: bool = False, # just appearance-based similarity or added OCR similarity
        verbose_show: int = 0, # 0 = nothing. 1 = show all screens w top interactable. k = show all; show user2 top-k predicted interactables
        savedir: str = None,
    ) -> bbox_type:
        bbox_clicked_user1 = self.get_interactable_at_click(image1, action_on_image1)
        best_bbox_user2, top_k_bbox_user2 = self.interactable_matching(
            image1, 
            image2, 
            action_on_image1, 
            mode,
            include_ocr=include_ocr,
            show_user1=(verbose_show > 0),
            ret_top_k=verbose_show,
        )
        
        if verbose_show == 1:
            self.show(image1, bbox_clicked_user1, "green", savedir=savedir+"user1img.png")
        if verbose_show > 1:
            self.show(image1, bbox_clicked_user1, "green", savedir=savedir+"user1img.png")
            color2 = ["blue"] * verbose_show
            color2[0] = "red"
            self.show(image2, top_k_bbox_user2, color2, savedir=savedir+f"user2pred_top{verbose_show}.png")

        return best_bbox_user2