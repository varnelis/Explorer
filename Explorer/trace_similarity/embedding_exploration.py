import os
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision import transforms
#from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from itertools import combinations
from collections import OrderedDict

from Explorer.overlay.shortlister import Shortlister


class Encoder():

    def __init__(self):
        ''' DEFINE MODEL '''
        FINETUNE_CLASSES = 2
        lr = 0.08

        shortlister = Shortlister()
        shortlister.set_model("interactable-detector").load_shortlister_model()
        self.khan_model = shortlister.shortlister_model

        self.khan_model.hparams.lr = lr
        self.khan_model.hparams.num_classes = FINETUNE_CLASSES
        mod = self.khan_model.model.head.classification_head
        mod.num_classes = FINETUNE_CLASSES

        self.embedding_distances = {}

        # Image transforms for input to Interactable Detector model
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 320
        max_size = 640
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.tensor_transform = transforms.ToTensor()

    def encoder(self, image: Image.Image, show_image: bool = False) -> list[torch.Tensor]:
        image = image.convert("RGB")
        image = self.tensor_transform(image)
        
        if show_image is True:
            transforms.functional.to_pil_image(image).show()
        image = self.transform([image])
        if show_image is True:
            transforms.functional.to_pil_image(image.tensors[0]).show()
        
        # get the features from the backbone -- C3-C5, P3-P7 backbone & FPN
        feature_embeddings = self.khan_model.model.backbone(image[0].tensors)
        if isinstance(feature_embeddings, torch.Tensor):
            feature_embeddings = OrderedDict([("0", feature_embeddings)])

        f_emb_sq = []
        for f in feature_embeddings.values():
            f = f.squeeze()
            f_emb_sq.append(f)

        return f_emb_sq

    def centerness(self, E: list[torch.Tensor]) -> list[torch.Tensor]:
        center = []
        for e in E:
            center_e = self.khan_model.model.head.regression_head.bbox_ctrness(e)
            center.append(center_e)
        return center

    def upsample_aggregate(self, E: list[torch.Tensor]) -> torch.Tensor:
        # upsample P4-P7 to P3 shape (256, 40, 44)
        upsample = torch.nn.Upsample(size=(40,44), mode='nearest')
        for e in range(len(E)):
            E[e] = upsample(E[e].unsqueeze(0))
        # concatenate to same tensor
        E_cat = torch.cat(E, dim=0)
        
        return E_cat.detach()
    
    def encoding_distance(self, e1: torch.Tensor, e2: torch.Tensor) -> float:
        e_diff = torch.abs(e1-e2)
        if len(e_diff.shape) == 1:
            return torch.linalg.norm(e_diff, dim=-1).item()
        mat_norm = torch.linalg.matrix_norm(e_diff) # frobenius matrix norm of abs diff
        return torch.linalg.norm(mat_norm).item() # frobenius vector norm
    
    '''
    def compare_encodings(self, idx1, idx2):
        fpn1 = self.encoder(idx1)
        fpn2 = self.encoder(idx2)

        distances = []
        for i in range(len(fpn1)):
            e1, e2 = fpn1[i], fpn2[i]
            dist = self.encoding_distance(e1, e2)
            distances.append(dist)
        
        self.embedding_distances[f"{idx1}-{idx2}"] = distances

    def compare_all_enc_combinations(self):
        top = self.khan_test_dataloader.__len__()
        combs = list(combinations(list(np.arange(top)), 2))

        for c in tqdm(combs[:5]):
            (idx1, idx2) = c
            self.compare_encodings(idx1, idx2)

        print(self.embedding_distances)

        with open('./embedding_distances.json', 'w', encoding='utf8') as f:
            json.dump(self.embedding_distances, f)

    def plot_all_enc_combinations(self):
        with open('./embedding_distances.json', 'r') as f:
            enc_d = json.load(f)

        top = self.khan_test_dataloader.__len__()
        heatmap = [
            np.zeros(shape=(top, top)),
            np.zeros(shape=(top, top)),
            np.zeros(shape=(top, top)),
            np.zeros(shape=(top, top)),
            np.zeros(shape=(top, top))
        ]

        for k in tqdm(enc_d):
            img_idxs = k.split('-')
            idx1 = int(img_idxs[0])
            idx2 = int(img_idxs[1])

            for i in range(5):
                heatmap[i][idx1,idx2] = enc_d[k][i]
                heatmap[i][idx2,idx1] = enc_d[k][i]

        P = 3
        for h in heatmap:
            fig, ax = plt.subplots(figsize=(15,15))
            im = ax.imshow(h)
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.title(f'Norm Comparison of All Test Image Embeddings from FPN P{P}', fontdict={'fontsize':30})
            plt.xlabel('Test Images', fontdict={'fontsize':25})
            fig.tight_layout()
            plt.savefig(f'./embedding_compare/all_emb_comp_P{P}.png')

            P += 1 # FPN feature level (P3-P7)
    '''

if __name__ == '__main__':
    
    enc = Encoder()

    '''
    E1 = enc.encoder(119, show_image=True)
    E2 = enc.encoder(17, show_image=True)
    E3 = enc.encoder(187, show_image=True)
    E4 = enc.encoder(120, show_image=True)
    E5 = enc.encoder(215, show_image=True)
    E6 = enc.encoder(91, show_image=True)

    print(enc.encoding_distance(E1[-1], E2[-1]))
    print(enc.encoding_distance(E3[-1], E4[-1]))
    print(enc.encoding_distance(E5[-1], E6[-1]))
    '''

    # Ablation
    img_base = './embedding_ablation/4461a1b2d7e048a08349f5f55fa23c07.png'
    img_white = './embedding_ablation/4461a1b2d7e048a08349f5f55fa23c07_mask_white.png'
    img_black = './embedding_ablation/4461a1b2d7e048a08349f5f55fa23c07_mask_black.png'

    E = enc.encoder(load_from_path=img_base)
    E_w = enc.encoder(load_from_path=img_white)
    E_b = enc.encoder(load_from_path=img_black)

    for i in range(5):
        f1, f2 = E_b[i], E[i]
        print(enc.encoding_distance(f1,f2))
