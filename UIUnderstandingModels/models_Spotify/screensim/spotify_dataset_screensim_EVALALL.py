import torch
import json
import random
import os
from PIL import Image
from tqdm import tqdm
from random import choices
#from itertools import combinations
from torchvision import transforms
import pytorch_lightning as pl
from embedding_exploration import Encoder

def random_viewport_from_full_EVAL(height, w, h):
    h1 = int(random.random() * (h - height))
    h2 = h1 + height
    viewport = (0, h1, w, h2)
    return viewport

def random_viewport_pair_from_full_EVAL(img_full, height_ratio):
    # print(img_full)
    img_pil = Image.open(img_full).convert("RGB")
    w, h = img_pil.size
    height = int(w * height_ratio)
    viewport1 = random_viewport_from_full_EVAL(height, w, h)
    vh1 = viewport1[1]
    delta = int(random.random() * (2 * height)) - height
    vh2 = vh1 + delta
    vh2 = min(max(0, vh2), h - height)
    viewport2 = (0, vh2, w, vh2 + height)
    view1 = img_pil.crop(viewport1)
    view2 = img_pil.copy().crop(viewport2)
    return (view1, view2)

class SpotifySimilarityDataset_EVAL(torch.utils.data.Dataset):
    def __init__(self, 
                 split_file="../../metadata/screensim/train_split_spotify_screensim.json", 
                 root_dir="../../downloads/spotify/screenshots", 
                 domain_map_file="../../metadata/screensim/domain_map_spotify.json",
                 ):
        super(SpotifySimilarityDataset_EVAL, self).__init__()
        
        self.root_dir = root_dir
        self.split_file = split_file
        
        # filter by split file
        with open(split_file, "r") as f:
            split_dict = json.load(f)
        split_set = set(split_dict['items'])
        #split_set = set([str(s) for s in split_list])
        
        with open(domain_map_file, "r") as f:
            self.domain_map = json.load(f)
            
        self.domain_list = []
        self.all_UUIDs = {} # uuid: group url
        for url in tqdm(self.domain_map):
            #if all([uuid in split_set for uuid in self.domain_map[url]]) and len(set(self.domain_map[url])) > 1:
            if url in split_set: ################## and len(set(self.domain_map[url])) > 1:
                self.domain_list.append(url)
                for _uuid in self.domain_map[url]:
                    self.all_UUIDs[_uuid] = url

        self.enc = Encoder()

        self.all_pair_UUIDs = self.all_combination_pairs()
        #list(combinations(self.all_UUIDs, 2))
            
        '''with open(duplicate_map_file, "r") as f:
            self.duplicate_map = json.load(f)
            
        self.duplicate_list = []
        for dn in tqdm(self.duplicate_map):
            if all([url in split_set for url in self.duplicate_map[dn]]):
                self.duplicate_list.append(dn)
                
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        ignore_ids = set()
        for ignore_file in uda_ignore_id_files:
            with open(ignore_file, "r") as f:
                ignore_file_ids = set(json.load(f))
                ignore_ids |= ignore_file_ids
        
        self.uda_dir = uda_dir
        self.uda_files = [f for f in os.listdir(uda_dir) if f.endswith(".jpg") and f.replace(".jpg", "") not in ignore_ids]
        '''

    def all_combination_pairs(self):
        comb_pairs = {}
        for uuid1 in self.all_UUIDs:
            for uuid2 in self.all_UUIDs:
                if uuid1 != uuid2:
                    group1 = self.all_UUIDs[uuid1]
                    group2 = self.all_UUIDs[uuid2]

                    if group1 == group2 and len(self.domain_map[group1])==1:
                        continue
                    elif group1 == group2:
                        comb_pairs[(uuid1,uuid2)] = 1 # label 1 = same
                    else:
                        comb_pairs[(uuid1,uuid2)] = 0 # label 0 = diff

        print("**************************************************")
        print("split file: ", self.split_file)
        print("Num UUIDs: ", len(self.all_UUIDs))
        print("Num Combinations: ", len(comb_pairs))
        print("**************************************************\n")

        return comb_pairs

    def sample_same_screen(self):
        try:
            with open(self.uuid2int_file, "r") as f:
                self.uuid2int = json.load(f)

            # randomly choose a duplicated URL
            random_url = random.choice(self.domain_list)
            [random_uuid_1, random_uuid_2] = random.sample(self.domain_map[random_url], 2)
            img1_path = os.path.join(self.root_dir, random_uuid_1 + ".png")
            img2_path = os.path.join(self.root_dir, random_uuid_2 + ".png")
            
            #pil_img1 = Image.open(img1_path).convert("RGB")
            #pil_img2 = Image.open(img2_path).convert("RGB")
            
            encodings_img1 = self.enc.encoder(load_from_path=img1_path)
            encodings_img2 = self.enc.encoder(load_from_path=img2_path)

            center_img1 = self.enc.centerness(encodings_img1)
            center_img2 = self.enc.centerness(encodings_img2)  
            
            encodings_img1_upsample = self.enc.upsample_aggregate(encodings_img1)
            encodings_img2_upsample = self.enc.upsample_aggregate(encodings_img2)

            center_img_1_upsample = self.enc.upsample_aggregate(center_img1)
            center_img_2_upsample = self.enc.upsample_aggregate(center_img2)

            # UUIDs for TF-IDF
            uuid_int1 = self.uuid2int[random_uuid_1]
            uuid_int2 = self.uuid2int[random_uuid_2]

            # ocr embeddings from Sentence-BERT (Sentence Transformers)
            #ocr1 = torch.tensor(self.uuid2ocr[''.join(random_uuid_1.split('-'))][1])
            #ocr2 = torch.tensor(self.uuid2ocr[''.join(random_uuid_2.split('-'))][1])

            return encodings_img1_upsample, encodings_img2_upsample, center_img_1_upsample, center_img_2_upsample#, uuid_int1, uuid_int2
        except:
            return self.sample_same_screen()

    def sample_different_screen(self): # different domain
        try:
            #with open(self.uuid2int_file, "r") as f:
            #    self.uuid2int = json.load(f)

            sampled_domains = random.sample(list(self.domain_map), 2)
            domain1 = sampled_domains[0]
            domain2 = sampled_domains[1]
            uuid1 = random.choice(self.domain_map[domain1])
            uuid2 = random.choice(self.domain_map[domain2])

            img1_path = os.path.join(self.root_dir, uuid1 + ".png")
            img2_path = os.path.join(self.root_dir, uuid2 + ".png")
            
            #pil_img1 = Image.open(img1_path).convert("RGB")
            #pil_img2 = Image.open(img2_path).convert("RGB")
            
            encodings_img1 = self.enc.encoder(load_from_path=img1_path)
            encodings_img2 = self.enc.encoder(load_from_path=img2_path)

            center_img1 = self.enc.centerness(encodings_img1)
            center_img2 = self.enc.centerness(encodings_img2)
            
            encodings_img1_upsample = self.enc.upsample_aggregate(encodings_img1)
            encodings_img2_upsample = self.enc.upsample_aggregate(encodings_img2)

            center_img_1_upsample = self.enc.upsample_aggregate(center_img1)
            center_img_2_upsample = self.enc.upsample_aggregate(center_img2)

            # UUIDs for TF-IDF
            #uuid_int1 = self.uuid2int[uuid1]
            #uuid_int2 = self.uuid2int[uuid2]

            # ocr embeddings from Sentence-BERT (Sentence Transformers)
            #ocr1 = torch.tensor(self.uuid2ocr[''.join(uuid1.split('-'))][1])
            #ocr2 = torch.tensor(self.uuid2ocr[''.join(uuid2.split('-'))][1])

            return encodings_img1_upsample, encodings_img2_upsample, center_img_1_upsample, center_img_2_upsample#, uuid_int1, uuid_int2
        except:
            return self.sample_different_screen()
            
    #def __iter__(self):
    #    while True:
    #        if random.random() <= 0.5:
    #            res = self.sample_same_screen()
    #            label = 1
    #        else:
    #            res = self.sample_different_screen()
    #            label = 0
    #        
    #        yield {'label': label, 'fpn1': res[0], 'fpn2': res[1], 'center1': res[2], 'center2': res[3], 'uuid_int1': res[4], 'uuid_int2': res[5]}
    
    def __len__(self):
        return len(self.all_pair_UUIDs)

    def __getitem__(self, idx):
        def return_next():  # for debugging
            return self.__getitem__(idx + 1)

        #with open(self.uuid2int_file, "r") as f:
        #    self.uuid2int = json.load(f)

        try:
            labelled_pairs = list(self.all_pair_UUIDs.keys())
            (uuid1, uuid2) = labelled_pairs[idx]
            label = self.all_pair_UUIDs[(uuid1,uuid2)]

            img1_path = os.path.join(self.root_dir, uuid1 + ".png")
            img2_path = os.path.join(self.root_dir, uuid2 + ".png")

            #pil_img1 = Image.open(img1_path).convert("RGB")
            #pil_img2 = Image.open(img2_path).convert("RGB")

            encodings_img1 = self.enc.encoder(load_from_path=img1_path)
            encodings_img2 = self.enc.encoder(load_from_path=img2_path)

            center_img1 = self.enc.centerness(encodings_img1)
            center_img2 = self.enc.centerness(encodings_img2)

            encodings_img1_upsample = self.enc.upsample_aggregate(encodings_img1)
            encodings_img2_upsample = self.enc.upsample_aggregate(encodings_img2)

            center_img_1_upsample = self.enc.upsample_aggregate(center_img1)
            center_img_2_upsample = self.enc.upsample_aggregate(center_img2)

            # UUIDs for TF-IDF
            #uuid_int1 = self.uuid2int[uuid1]
            #uuid_int2 = self.uuid2int[uuid2]

            # ocr embeddings from Sentence-BERT (Sentence Transformers)
            #ocr1 = torch.tensor(self.uuid2ocr[''.join(uuid1.split('-'))][1])
            #ocr2 = torch.tensor(self.uuid2ocr[''.join(uuid2.split('-'))][1])

            return {'label':label, 'fpn1':encodings_img1_upsample, 'fpn2':encodings_img2_upsample, 'center1':center_img_1_upsample, 'center2':center_img_2_upsample}#, uuid_int1, uuid_int2#, ocr1, ocr2

        except Exception as e:
            print('failed ', idx, str(e))
            return return_next()

class SpotifySimilarityDataModule_EVAL(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=2, split_file="../../metadata/screensim/train_split_spotify_screensim.json"):
        super(SpotifySimilarityDataModule_EVAL, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_file = split_file
        
        self.train_dataset = SpotifySimilarityDataset_EVAL(split_file=split_file)
        self.val_dataset = SpotifySimilarityDataset_EVAL(split_file="../../metadata/screensim/val_split_spotify_screensim.json")
        self.test_dataset = SpotifySimilarityDataset_EVAL(split_file="../../metadata/screensim/test_split_spotify_screensim.json")
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True)
