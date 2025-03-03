import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np


class WikipediaUIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="../../downloads/wikipedia/screenshots",
        class_dict_path="../../metadata/screenrecognition/class_map_wikipedia_manual.json",
        id_list_path="../../metadata/screenrecognition/train_ids_wikipedia.json",
        track_uuid=False,
    ):
        with open(id_list_path, "r") as f:
            id_dict = json.load(f)
        self.id_list = id_dict['items']

        self.root = root
        self.img_transforms = transforms.ToTensor()

        with open(class_dict_path, "r") as f:
            class_dict = json.load(f)

        self.idx2Label = class_dict["idx2Label"]
        self.label2Idx = class_dict["label2Idx"]

        self.track_uuid = track_uuid

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        def return_next():  # for debugging
            return WikipediaUIDataset.__getitem__(self, idx + 1)

        try:
            img_path = os.path.join(self.root, self.id_list[idx]['uuid'])
            img_path += '.png' # add extension; id_list only includes uuid, not extension

            pil_img = Image.open(img_path).convert("RGB")
            img = self.img_transforms(pil_img)

            # get annotations dictionary with bboxes
            with open(
                img_path.replace(".png", ".json").replace("screenshots", "annotations"),
                "r",
            ) as root_file:
                annotations = json.load(root_file)

            # get bounding box coordinates for each mask
            boxes = []
            for b in annotations["clickable"]:  # list of dicts (for each bbox) output
                boxes.append(b['bbox'])
            labels = [self.label2Idx["clickable"]] * len(boxes)

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            # iscrowd = torch.zeros((num_labels,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            
            # track tensor to PIL image conversion to identify mAP per test image
            if self.track_uuid:
                try:
                    with open('tensor_to_img.json','r') as f:
                        t2img = json.load(f)
                except:
                    t2img = {}
                t2img[idx] = self.id_list[idx]['uuid']
                with open('tensor_to_img.json','w') as f:
                    json.dump(t2img, f)

            return img, target

        except Exception as e:
            print("failed", idx, self.id_list[idx]['uuid'], str(e))
            return return_next()


class WikipediaUIOneHotLabelDataset(WikipediaUIDataset):
    def __getitem__(self, idx):
        img, res_dict = super(WikipediaUIOneHotLabelDataset, self).__getitem__(idx)
        num_classes = 2
        one_hot_labels = F.one_hot(res_dict["labels"], num_classes=num_classes)
        res_dict["labels"] = one_hot_labels
        return img, res_dict


# https://github.com/pytorch/vision/blob/5985504cc32011fbd4312600b4492d8ae0dd13b4/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))


class WikipediaUIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=2, one_hot_labels=True, subset_size=-1, rand_seed=0):
        ''' Dataloader for Wikipedia data at given batch size. 2 Workers works good on JADE. If subset_size=-1, all; else subset. '''
        super(WikipediaUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.rand_seed = rand_seed

        if one_hot_labels:
            self.train_dataset = WikipediaUIOneHotLabelDataset(
                id_list_path="../../metadata/screenrecognition/train_ids_wikipedia.json"
            )
            self.val_dataset = WikipediaUIOneHotLabelDataset(
                id_list_path="../../metadata/screenrecognition/val_ids_wikipedia.json"
            )
            self.test_dataset = WikipediaUIOneHotLabelDataset(
                id_list_path="../../metadata/screenrecognition/test_ids_wikipedia.json"
            )
        else:
            self.train_dataset = WikipediaUIDataset(
                id_list_path="../../metadata/screenrecognition/train_ids_wikipedia.json"
            )
            self.val_dataset = WikipediaUIDataset(
                id_list_path="../../metadata/screenrecognition/val_ids_wikipedia.json"
            )
            self.test_dataset = WikipediaUIDataset(
                id_list_path="../../metadata/screenrecognition/test_ids_wikipedia.json"
            )

    def train_dataloader(self):
        if self.subset_size != -1:
            print("\nWikipediaUIDataModule: SUBSET SIZE != -1, RAND DATALOADER SUBSIZE\n")
            np.random.seed(self.rand_seed)
            subset_indices = np.random.choice(np.arange(0,len(self.train_dataset)), self.subset_size, replace=False)
            subset_dataset = torch.utils.data.Subset(self.train_dataset, list(subset_indices))
            return torch.utils.data.DataLoader(
                subset_dataset,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
            )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )
