import sys
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ui_datasets import *
from tqdm import tqdm
from typing import List, Dict, Literal
from pprint import pprint

class Benchmark_Web350k(object):

    def __init__(self, test_data):
        super().__init__()
        
        self.model = torch.jit.load("../../downloads/checkpoints/screenrecognition-web350k.torchscript") # Web350k -- not VINS finetune
        self.num_classes: int = 32

        self.test_data = test_data

        #self.images: List = []
        #self.image_file = './web350k_test_pred/images.pt'
        #try:
        #    self.images = torch.load(self.image_file)
        #    self.targets = torch.load(self.target_file)
        #except FileNotFoundError:
        #    self._getItems()
        #print('Images & Targets loaded...\n')

        self.preds: List = []
        self.targets: List = []
        self.pred_file: str = './web350k_test_pred/preds.pt'
        self.target_file: str = './web350k_test_pred/targets.pt'

        self.next_batch: int = 0
        self.next_batch_file: str = './web350k_test_pred/next_batch.json'

    def _getItems(self) -> None:

        for i in range(0, len(self.test_data.keys), 100):
            start = i
            end = i+100 if i+100<len(self.test_data.keys) else len(self.test_data.keys)

            for idx in range(start, end):
                img, target = self.test_data.__getitem__(idx)
            
            try: del target['image_id'] # remove this dict key to agree with mAP package input
            except: pass
            
            self.images.append(img)
            self.targets.append(target)

        torch.save(self.images, self.image_file)
        torch.save(self.targets, self.target_file)
    
    def _getSinglePredictionGroundTruth(self, index) -> None:
        
        img_input, target = self.test_data.__getitem__(index)

        output = self.model([img_input])[1] # get pred
        # pred -> add to pred array
        pred_box = output[0]['boxes']
        pred_score = output[0]['scores']
        pred_label = output[0]['labels']

        self.preds.append({
            'boxes': pred_box,
            'scores': pred_score,
            'labels': pred_label,
        })

        #self.targets_original = target

        labels, cat_labels = [], []
        cat_boxes = []
        for l in range(len(target['labels'])):
            _multiclass = False
            for i in range(target['labels'].shape[1]):
                if target['labels'][l][i] and not _multiclass:
                    labels.append(i)
                    _multiclass = True
                elif target['labels'][l][i] and _multiclass:
                    cat_labels.append(i)
                    cat_boxes.append(torch.clone(target['boxes'][l]).tolist())
        target = dict(
            boxes = torch.cat((target['boxes'], torch.tensor(cat_boxes)), dim=0), # updated boxes
            labels = torch.tensor(labels + cat_labels) # updated labels
        )
        self.targets.append(target)

    def getPred(self, iter: int = 0, batch_len: Literal['all'] | int = 10) -> bool:

        try:
            with open(self.next_batch_file, 'r') as f:
                self.next_batch = json.load(f)
            if self.next_batch: # if not 0 (else start with empty preds/gts)
                pred_existing = torch.load(self.pred_file)
                target_existing = torch.load(self.target_file)
                self.preds.extend(pred_existing)
                self.targets.extend(target_existing)
        except FileNotFoundError:
            print('No existing pred/gts file. Move on to compute...')

        batch_start = self.next_batch
        if batch_len == 'all':
            batch_end = len(self.test_data.keys)
            stop = True
        else:
            if self.next_batch + batch_len < len(self.test_data.keys):
                batch_end = self.next_batch + batch_len 
                stop = False
            else:
                batch_end = len(self.test_data.keys)
                stop = True

        for i in tqdm(range(batch_start, batch_end), desc=f'Batch {[batch_start, batch_end]}'):
            self._getSinglePredictionGroundTruth(i)

        # saving to files
        torch.save(self.preds, self.pred_file)
        torch.save(self.targets, self.target_file)
        #torch.save(self.targets_original, './web350k_test_pred/target_original.pt')
        with open(self.next_batch_file, 'w') as f:
            json.dump(batch_end, f)

        return stop

    def compute_mAP_score(self, map_type: Literal ['COCO', 'Pascal-VOC'] = 'COCO', thresholds: List = [0.5]) -> Dict:

        if len(self.preds) == 0 and len(self.targets) == 0:
            print('Loading into preds & targets...')
            try:
                pred_existing = torch.load(self.pred_file)
                target_existing = torch.load(self.target_file)
                self.preds.extend(pred_existing)
                self.targets.extend(target_existing)
            except FileNotFoundError:
                print('No existing pred file. Compute predictions first...')
                sys.exit(0)

        if map_type == 'COCO':
            MAP = MeanAveragePrecision(box_format="xyxy", iou_thresholds=None, class_metrics=True, average='macro')
        elif map_type == 'Pascal-VOC':
            MAP = MeanAveragePrecision(box_format="xyxy", iou_thresholds=thresholds, class_metrics=True, average='macro')

        assert(len(self.preds)==len(self.targets))
        for i in tqdm(range(len(self.preds)),desc='MAP iter'):
            MAP.update([self.preds[i]], [self.targets[i]])

        metrics = MAP.compute()

        if map_type == 'COCO': print('\nCOCO mAP@[0.5,0.05,0.95]')
        elif map_type == 'Pascal-VOC': print(f'\nPascal VOC mAP@{thresholds}')
        pprint(metrics)

        for key in metrics.keys():
            if isinstance(metrics[key], torch.Tensor):
                metrics[key] = metrics[key].tolist()
        save_file = './web350k_test_pred/MAP_Web350k'
        with open(save_file, 'w') as sf:
            json.dump(metrics, sf)

        return metrics


def get_pred_target(n):
    path: str = 'D:/Iason Chaimalas/Iason Chaimalas UCL/UCL Academic 2023-24/Yr4Proj/CODE/WebUI-Test-Data-DownloadExtract/'
    test_data = WebUIDataset(split_file = path + 'test_split_webui.json',
                             rawdata_screenshots_dir = path + 'ds')
    print('Test Data computed...\n')

    rand_batches = np.random.choice(a=len(test_data.keys)-10, size=(1,n), replace=False)

    stop = False
    i = 0
    #while not stop:
    for _ in range(n):
        with open('./web350k_test_pred/next_batch.json', 'w') as fp:
            json.dump(int(rand_batches[0,i]), fp) # set random batch to use
        b = Benchmark_Web350k(test_data=test_data)
        stop = b.getPred(iter=i, batch_len=10)
        i += 1


if __name__ == '__main__':

    #get_pred_target(50)

    m = Benchmark_Web350k(test_data=None)
    metrics = m.compute_mAP_score(map_type='Pascal-VOC')
    metrics = m.compute_mAP_score(map_type='COCO')
