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

from ui_models_distributed import *


class Encoder():

    def __init__(self):
        ''' DEFINE MODEL '''
        FINETUNE_CLASSES = 2
        class_weights = [0,1]
        lr = 0.08
        #freeze_support()

        # Load trained Interactable Detector to use its Feature-Pyramid features
        # for Screen Similarity model training
        self.khan_model = UIElementDetector.load_from_checkpoint(
            "khan_interactable_best.ckpt", 
            val_weights=class_weights, 
            test_weights=class_weights, 
            lr=lr,
        )
        self.khan_model.hparams.lr = lr
        self.khan_model.hparams.num_classes = FINETUNE_CLASSES
        mod = self.khan_model.model.head.classification_head
        mod.num_classes = FINETUNE_CLASSES

        self.dataloading()

        self.tensor_imgs = {}
        self.embedding_distances = {}

    def dataloading(self):
        ''' GET IMAGE '''
        #self.khan_datamodule = KhanUIDataModule(batch_size=1, num_workers=1)
        #self.khan_test_dataloader = self.khan_datamodule.test_dataset  #loader()
        #images = torch.tensor(images)

        ''' TRANSFORM IMAGE '''
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 320
        max_size = 640
        
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.tensor_transform = transforms.ToTensor()

    def encoder(self, data_idx: int = 0, show_image: bool = False, load_from_path = None):
        
        if load_from_path:
            images = Image.open(load_from_path).convert("RGB")
            images = self.tensor_transform(images)
        else:
            raise NotImplementedError
            #images, _ = self.khan_test_dataloader.__getitem__(data_idx)
            #next(iter(khan_test_dataloader))

        if show_image:
            transforms.functional.to_pil_image(images).save(f'./InputTransforms_GeneralisedRCNN/before_transform_idx{data_idx}.png')
        
        images = self.transform([images])
        if show_image:
            transforms.functional.to_pil_image(images.tensors[0]).save(f'./InputTransforms_GeneralisedRCNN/after_transform_idx{data_idx}.png')
        
        self.tensor_imgs[data_idx] = images[0].tensors # for visualisation

        ''' GET EMBEDDINGS '''
        # get the features from the backbone -- C3-C5, P3-P7 backbone & FPN
        feature_embeddings = self.khan_model.model.backbone(images[0].tensors)
        if isinstance(feature_embeddings, torch.Tensor):
            feature_embeddings = OrderedDict([("0", feature_embeddings)])

        f_emb_sq = []
        for f in feature_embeddings.values():
            f = f.squeeze()
            f_emb_sq.append(f)

        #head_outputs = khan_model.model.head(feature_embeddings)

        return f_emb_sq

    def centerness(self, E):
        center = []
        for e in E:
            center_e = self.khan_model.model.head.regression_head.bbox_ctrness(e)
            center.append(center_e)
        return center

    def upsample_aggregate(self, E):
        # upsample P4-P7 to P3 shape (256, 40, 44)
        upsample = torch.nn.Upsample(size=(40,44), mode='nearest')
        for e in range(len(E)):
            E[e] = upsample(E[e].unsqueeze(0))
        # concatenate to same tensor
        E_cat = torch.cat(E, dim=0)
        
        return E_cat.detach()
    
    def encoding_distance(self, e1, e2):
        e_diff = torch.abs(e1-e2)
        mat_norm = torch.linalg.matrix_norm(e_diff) # frobenius matrix norm of abs diff
        return torch.linalg.norm(mat_norm) # frobenius vector norm
    
    def compare_encodings(self, idx1, idx2):
        fpn1 = self.encoder(idx1)
        fpn2 = self.encoder(idx2)

        distances = []
        for i in range(len(fpn1)):
            e1, e2 = fpn1[i], fpn2[i]
            dist = self.encoding_distance(e1, e2)
            distances.append(dist.item())
        
        self.embedding_distances[f"{idx1}-{idx2}"] = distances

    def tsne_encoding(self, idx):
        raise NotImplementedError

        #fpn = self.encoder(idx)

        #fpn_tsne = TSNE(n_components=3, perplexity=30, learning_rate=50).fit_transform(
        #    fpn[0].detach().cpu().numpy()
        #)
        #print(type(fpn_tsne))
        #print(fpn_tsne.shape)

    def show_image(self, data_idx):
        img = self.tensor_imgs[data_idx]

        img_pil = transforms.functional.to_pil_image(img[0])
        #img_pil = Image.open(img_pil)
        img_pil.show()

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
        heatmap = [np.zeros(shape=(top, top)),
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
