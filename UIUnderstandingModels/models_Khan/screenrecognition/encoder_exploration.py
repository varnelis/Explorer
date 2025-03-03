import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms.functional import to_pil_image
#from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import json
import os
from itertools import combinations
from collections import OrderedDict
from ui_dataset_khan import *
from ui_models_distributed import *


class Encoder():

    def __init__(self):
        ''' DEFINE MODEL '''
        FINETUNE_CLASSES = 2
        class_weights = [0,1]
        lr = 0.08
        #freeze_support()

        ARTIFACT_DIR = "./checkpoints_screenrecognition_khan"
        self.khan_model = UIElementDetector.load_from_checkpoint(
            os.path.join(ARTIFACT_DIR, "last-v8.ckpt"), 
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
        self.khan_datamodule = KhanUIDataModule(batch_size=1, num_workers=1)
        self.khan_test_dataloader = self.khan_datamodule.test_dataset  #loader()
        #images = torch.tensor(images)

        ''' TRANSFORM IMAGE '''
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 320
        max_size = 640
        
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def encoder(self, data_idx: int = 0, show_image: bool = False):
        images, targets = self.khan_test_dataloader.__getitem__(data_idx)
            #next(iter(khan_test_dataloader))
        if show_image:
            to_pil_image(images).show()
        
        images, targets = self.transform([images], [targets])
        if show_image:
            to_pil_image(images.tensors[0]).show()
        
        self.tensor_imgs[data_idx] = images.tensors # for visualisation

        ''' GET EMBEDDINGS '''
        # get the features from the backbone -- C3-C5, P3-P7 backbone & FPN
        feature_embeddings = self.khan_model.model.backbone(images.tensors)
        if isinstance(feature_embeddings, torch.Tensor):
            feature_embeddings = OrderedDict([("0", feature_embeddings)])

        f_emb_sq = []
        for f in feature_embeddings.values():
            f = f.squeeze()
            f_emb_sq.append(f)

        #head_outputs = khan_model.model.head(feature_embeddings)

        return f_emb_sq

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
        pass

        #fpn = self.encoder(idx)

        #fpn_tsne = TSNE(n_components=3, perplexity=30, learning_rate=50).fit_transform(
        #    fpn[0].detach().cpu().numpy()
        #)
        #print(type(fpn_tsne))
        #print(fpn_tsne.shape)

    def show_image(self, data_idx):
        img = self.tensor_imgs[data_idx]

        img_pil = to_pil_image(img[0])
        #img_pil = Image.open(img_pil)
        img_pil.show()


if __name__ == '__main__':
    
    enc = Encoder()

    top = enc.khan_test_dataloader.__len__()
    combs = list(combinations(list(np.arange(top)), 2))

    for c in tqdm(combs):
        (idx1, idx2) = c
        enc.compare_encodings(idx1, idx2)

    print(enc.embedding_distances)

    with open('./embedding_distances.json', 'w', encoding='utf8') as f:
        json.dump(enc.embedding_distances, f)
    
    
    #enc.show_image(idx1)
    #enc.show_image(idx2)
    #enc.tsne_encoding(idx1)
