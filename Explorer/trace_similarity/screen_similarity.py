import os
import gdown
import torch
import numpy as np
from PIL import Image
from Explorer.db.mongo_db import MongoDBInterface
from Explorer.trace_similarity.embedding_exploration import Encoder
from Explorer.trace_similarity.screensimilarity_model_CenterFPN import UIScreenEmbedder
from Explorer.ocr.ocr import KhanOCR

screensim_version = "v1.0.0"
model2file = {"screensimilarity": "screensimilarity.ckpt"}

class ScreenSimilarity(Encoder):

    def __init__(self):
        super().__init__()
        self.weights_dir = 'weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self._get_screensim_weights()
        self.screensim_model = UIScreenEmbedder.load_from_checkpoint(self.model_weights_path)
        self.ocr_model = KhanOCR()

    def _get_img_features(self, image: Image.Image):        
        fpn = self.encoder(image=image, show_image=False)
        center = self.centerness(fpn)
        fpn_upsample = self.upsample_aggregate(fpn)
        center_upsample = self.upsample_aggregate(center)
        return fpn_upsample, center_upsample
    
    def _get_screensim_weights(self) -> None:
        """ Get file for weights of the screensim model (local or from mongodb) """
        
        MongoDBInterface.connect()
        weights = MongoDBInterface.get_items({"version": screensim_version}, "screensimilarity").limit(1)
        weights = list(weights)[0]

        self.model_weights_path = self._gdown_url(weights["url"], model2file["screensimilarity"])

    def _gdown_url(self, url: str, model_key: str) -> None:
        outpath = os.path.join(self.weights_dir, model_key)
        if not os.path.exists(outpath):
            #url = url + '/view?usp=share_link'
            print('Retrieving weights from URL = ', url)
            gdown.download(url=url, output=outpath, fuzzy=True)
        return outpath

    def image2embedding(self, image: Image.Image) -> torch.Tensor:
        batch_size = 1
        fpn, center_upsample = self._get_img_features(image)
        fpn = torch.broadcast_to(fpn, (batch_size, 5, 256, 40, 44))
        center = torch.broadcast_to(center_upsample, (batch_size, 5, 256, 40, 44))
        embeddings = torch.cat((fpn.unsqueeze(1), center.unsqueeze(1)), dim=1)
        return embeddings
    
    def image2ocr(self, image: Image.Image) -> list[tuple[list[float, float, float, float], str]]:
        _, ocr_bbox_text = self.ocr_model.get_base_ocr(np.array(image), confidence=0.5, ret_bbox_plus_text=True)
        return ocr_bbox_text
    
    def ocr2embedding(self, ocr_text: list[str], concat: bool = False) -> torch.Tensor:
        ocr_embeddings = self.ocr_model.word_embedding(ocr_text, concat=concat)
        return torch.tensor(ocr_embeddings)

    def embeddings2similarity(self, embeddings_img1: torch.Tensor, embeddings_img2: torch.Tensor) -> tuple[float, bool]:
        outs1 = self.screensim_model.forward(embeddings_img1)
        outs2 = self.screensim_model.forward(embeddings_img2)

        dist = self.encoding_distance(outs1, outs2)
        thresh = 0.5 * (self.screensim_model.hparams.margin_pos + self.screensim_model.hparams.margin_neg)
        preds = dist < thresh

        return (dist, preds)
    
    def trace_self_similarity(self, trace_frames: list[Image.Image]) -> list[float]:
        """ Takes list of trace frame UUIDs/paths and outputs list of consecutive-frame similarity """
        distances = []
        for image1, image2 in zip(trace_frames, trace_frames[1:]):
            embeddings_img1 = self.image2embedding(image1)
            embeddings_img2 = self.image2embedding(image2)
            (dist, preds) = self.embeddings2similarity(embeddings_img1, embeddings_img2)
            distances.append(dist)
        #print('Distances: ', distances)
        return distances
    
    def ocr_embedding_similarity(self, ocr_embeddings_img1: torch.Tensor, ocr_embeddings_img2: torch.Tensor) -> float:
        ocr_dist = self.encoding_distance(ocr_embeddings_img1, ocr_embeddings_img2)
        return ocr_dist

def uuid2image(uuid: str) -> Image.Image:
    path = os.path.join('./selenium_scans/screenshots', uuid + '.png')
    image = Image.open(path)
    return image


if __name__ == '__main__':
    screensim = ScreenSimilarity()
    trace_frames = ['f81ae48ac9894eb79e8708abcb97843e', '20a58d760c4a4e619270a042c4eb451f']
    for i in range(len(trace_frames)):
        trace_frames[i] = uuid2image(trace_frames[i])
    screensim.trace_self_similarity(trace_frames, use_ocr=False)