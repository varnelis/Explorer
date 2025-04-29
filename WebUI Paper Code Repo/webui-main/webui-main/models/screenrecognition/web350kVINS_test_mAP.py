import sys
#sys.path.append("../../models/screenrecognition")

from typing import List, Dict, Literal
from tqdm import tqdm
from pprint import pprint
import json, xmltodict
import torch
import numpy
from PIL import Image, ImageDraw
from torchvision import transforms
from ui_models import *
from ui_datasets import *

from mean_average_precision import MetricBuilder
from torchmetrics.wrappers import BootStrapper
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Benchmark(object):

    def __init__(self, class_map_path: str, test_images_path: str, test_targets_path: str) -> None:
        
        super().__init__()

        self.model = torch.jit.load("../../downloads/checkpoints/screenrecognition-web350k-vins.torchscript")
        
        self.class_map_path: str = class_map_path
        self.images_path: str = test_images_path
        self.targets_path: str = test_targets_path

        # Get VINS label-to-index dict
        with open(self.class_map_path, "r") as f:
            class_map = json.load(f)
        self.label2Idx = class_map['label2Idx']

        # get image filepaths & corresponding annotations
        self.images: List = []
        self.targets: List = []
        self._getImagesTargets() # set image & target lists

        self.img_transforms = transforms.ToTensor()
        self.num_classes: int = 13

        self.preds: List = []
        self.gts: List = []
        self.pred_file: str = './vins_test_pred/preds.pt'
        self.gts_file: str = './vins_test_pred/gts.pt'

        self.next_batch: int = 0
        self.next_batch_file: str = './vins_test_pred/next_batch.json'

    def _getImagesTargets(self):
        
        # image paths relative to current dir
        with open(self.images_path, 'r') as f_img:
            self.images = json.load(f_img)
        
        # list of target dicts with annotations for each corresponding image
        with open(self.targets_path, 'r') as f_targ:
            self.targets = json.load(f_targ)

    def _getSinglePredictionAndGroundTruth(self, index) -> None:
        
        image_file = self.images[index]
        target_dict = self.targets[index]
        
        ########## PREDICTION ##########

        # load single image (& close afterward for memory efficiency)
        with Image.open(image_file) as test_image:
            img_input = self.img_transforms(test_image)
        
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
        ## This is alternative for mean_average_precision package -- gives bugs
        #self.preds.append(torch.cat((pred_box, pred_label.unsqueeze(-1), pred_score.unsqueeze(-1)), dim=-1))

        ########## TARGET / GROUND TRUTH ##########

        # get corresponding target dict -> extract annotated bounding box & true label
        gts_i_box = []
        gts_i_label = []
        target_len = len(target_dict['annotation']['object'])
        for i in range(target_len):
            xmin = target_dict['annotation']['object'][i]['bndbox']['xmin']
            ymin = target_dict['annotation']['object'][i]['bndbox']['ymin']
            xmax = target_dict['annotation']['object'][i]['bndbox']['xmax']
            ymax = target_dict['annotation']['object'][i]['bndbox']['ymax']
            target_box = torch.tensor([float(xmin), float(ymin), float(xmax), float(ymax)])
            target_label_str: str = target_dict['annotation']['object'][i]['name']
            target_label: int = self.label2Idx[target_label_str]
            
            #if len(target_label.shape) == 1:
            #    for ci in range(target_label.shape[0]):
            #        if target_label[ci] > 0:
            #            gts_i.append(torch.cat((target_box, torch.tensor([ci, 0, 0])), dim=-1))
            
            if target_label > 0:
                gts_i_box.append(target_box)
            else:
                gts_i_box.append(torch.zeros(0,4))
            gts_i_label.append(target_label)

            ## This is alternative for mean_average_precision package -- gives bugs
            #if target_label > 0:
            #    gts_i.append(torch.cat((torch.tensor(target_box), torch.tensor([target_label, 0, 0])), dim=-1))
        # add to gts array
        self.gts.append({
            'boxes': torch.stack(gts_i_box),
            'labels': torch.tensor(gts_i_label),
        })
        
        #self.gts.append(torch.stack(gts_i) if len(gts_i) > 0 else torch.zeros(0,7))
    
    def getPredGts(self, iter: int = 0) -> bool:

        try:
            with open(self.next_batch_file, 'r') as f:
                self.next_batch = json.load(f)

            if self.next_batch: # if not 0 (else start with empty preds/gts)
                pred_existing = torch.load(self.pred_file)
                gts_existing = torch.load(self.gts_file)

                self.preds.extend(pred_existing)
                self.gts.extend(gts_existing)


        except FileNotFoundError:
            print('No existing pred/gts file. Move on to compute...')

        batch_start = self.next_batch
        if self.next_batch + 10 < len(self.targets):
            batch_end = self.next_batch + 10 
            stop = False
        else:
            batch_end = len(self.targets)
            stop = True

        #print("\n***************************************")
        #print("********** PRED & GTS START ***********")
        #print("***************************************\n")

        for i in tqdm(range(batch_start, batch_end), desc=f'Iter {iter}; Pred/GT Len {len(self.preds)}'):
            self._getSinglePredictionAndGroundTruth(i)

        #print("\n***************************************")
        #print("*********** PRED & GTS END ************")
        #print("***************************************\n")

        # saving to files
        torch.save(self.preds, self.pred_file)
        torch.save(self.gts, self.gts_file)
        with open(self.next_batch_file, 'w') as f:
            json.dump(batch_end, f)

        return stop

    def compute_mAP_score(self, map_type: Literal ['COCO', 'Pascal-VOC'] = 'COCO', 
                          thresholds: List = [0.5],
                          std: bool = False) -> Dict:

        try:
            pred_existing = torch.load(self.pred_file)
            gts_existing = torch.load(self.gts_file)

            self.preds.extend(pred_existing)
            self.gts.extend(gts_existing)

        except FileNotFoundError:
            print('No existing pred/gts file. Compute them first...')
            sys.exit(0)

        # ignore predictions with score < threshold ... but this is unnecessary
        #if ignore_below_threshold:
        #    for p in range(len(self.preds)):
        #        pred = self.preds[p]
        #        boxes = pred['boxes']
        #        scores = pred['scores']
        #        labels = pred['labels']
        #
        #        boxes = boxes[scores > threshold]
        #        labels = labels[scores > threshold]
        #        scores = scores[scores > threshold]
        #        
        #        self.preds[p] = {'boxes': boxes, 'scores': scores, 'labels': labels}

        if map_type == 'COCO':
            MAP = MeanAveragePrecision(box_format="xyxy", iou_thresholds=None, class_metrics=True)
        elif map_type == 'Pascal-VOC':
            MAP = MeanAveragePrecision(box_format="xyxy", iou_thresholds=thresholds, class_metrics=True)

        assert(len(self.preds)==len(self.gts))

        if not std:
            for i in range(len(self.preds)):
                MAP.update([self.preds[i]], [self.gts[i]])
            
            metrics = MAP.compute()
        else:
            bMAP = BootStrapper(MAP)
            for i in range(len(self.preds)):
                bMAP.update([self.preds[i]], [self.gts[i]])
            metrics = bMAP.compute()

        if map_type == 'COCO': print('\nCOCO mAP@[0.5,0.05,0.95]')
        elif map_type == 'Pascal-VOC': print(f'\nPascal VOC mAP@{thresholds}')
        pprint(metrics)

        for key in metrics.keys():
            if isinstance(metrics[key], torch.Tensor):
                metrics[key] = metrics[key].tolist()
        save_file = './vins_test_pred/MAP_Web350kVINS'
        with open(save_file, 'w') as sf:
            json.dump(metrics, sf)

        return metrics
    
    def compute_mAP_score_buggy(self) -> None:
        
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=self.num_classes)
        
        assert(len(self.preds)==len(self.gts))
        for i in range(len(self.preds)):
            # detach from pred & gts (require grad)
            pred_d = self.preds[i].detach().cpu().numpy()
            gts_d = self.gts[i].detach().cpu().numpy()

            print('pred')
            print(pred_d, type(pred_d))
            print('gt')
            print(gts_d, type(gts_d))
            
            metric_fn.add(pred_d, gts_d)

        metrics = metric_fn.value(iou_thresholds=0.5)

        print('Metrics of finetuned WebUI-350k on VINS test dataset:\n')
        print(metrics)


class Benchmark_FullImgLoad(object):

    def __init__(self):
        super().__init__()

    def _vins_image_processing(self):
        with open('../../metadata/screenrecognition/test_ids_vins.json', 'r') as test_vins:
            vins_test_ids = list(map(lambda x: '../../downloads/vins/All Dataset' + x[1:], json.load(test_vins)))
        vins_test_labels = list(map(lambda x: x.replace('JPEGImages','Annotations').replace('jpg','xml'), vins_test_ids))

        # convert image path to tensor
        img_transforms = transforms.ToTensor()
        vins_test_imgs = list(map(lambda im: img_transforms(Image.open(im)), vins_test_ids))

        # xml annotation to dict for ground truth
        vins_test_gt = []
        for annotation in vins_test_labels:
            with open(annotation,'r',encoding='utf-8') as fp:
                ann_xml = fp.read()
            vins_test_gt.append(xmltodict.parse(ann_xml))

        # json save
        torch.save(vins_test_imgs, './vins_test_pred/vins_test_imgs.pt')
        with open('./vins_test_pred/vins_test_gt.json', 'w') as fp:
            json.dump(vins_test_gt, fp)

    def _vins_image_load(self):
        # load vins images & transform to Image obj tensor
        vins_test_imgs = torch.load('./vins_test_pred/vins_test_imgs.pt')
        with open('./vins_test_pred/vins_test_gt.json','r') as fp:
            vins_test_gt = json.load(fp)
        return [vins_test_imgs, vins_test_gt]

    def test_pred_gt(self):

        [images, targets] = self._vins_image_load()
        print('load OK............')

        # pretrained Web350k with VINS finetuning
        model = torch.jit.load("../../downloads/checkpoints/screenrecognition-web350k-vins.torchscript") # model
        #checkpoints = glob.glob("../../downloads/checkpoints/screenrecognition-web350k-vins.ckpt")
        #for checkpoint in tqdm(checkpoints):
        #    m = UIElementDetector.load_from_checkpoint(checkpoint).eval()

        class_map_file = "../../metadata/screenrecognition/class_map_vins_manual.json"
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        label2Idx = class_map['label2Idx']

        preds = []
        gts = []

        import time
        for batch in range(0,len(images),5):
            batch_start = batch
            batch_end = batch+5 if batch+5<len(images) else len(images)
            self.test_pred_gt_batched(preds, gts, images, targets, model, label2Idx, batch_start, batch_end)
            time.sleep(1)

        return preds, gts, model

    def test_pred_gt_batched(self, preds, gts, images, targets, m, label2Idx, batch_start, batch_end):
        outputs = m(images)[1]
        
        for batch_i in tqdm(range(batch_start, batch_end)):
            
            #try:
            #outputs = m([images[batch_i]])[0]
            #except:
            #    print(f'Skipped batch {batch_i}')
            #    continue
            
            pred_box = outputs[0]['boxes']
            pred_score = outputs[0]['scores']
            pred_label = outputs[0]['labels']
            
            preds.append(torch.cat((pred_box, pred_label.unsqueeze(-1), pred_score.unsqueeze(-1)), dim=-1))

            gts_i = []
            target_len = len(targets[batch_i]['annotation']['object'])
            for i in range(target_len):
                xmin = targets[batch_i]['annotation']['object'][i]['bndbox']['xmin']
                ymin = targets[batch_i]['annotation']['object'][i]['bndbox']['ymin']
                xmax = targets[batch_i]['annotation']['object'][i]['bndbox']['xmax']
                ymax = targets[batch_i]['annotation']['object'][i]['bndbox']['ymax']
                target_box = [float(xmin), float(ymin), float(xmax), float(ymax)]
                target_label_str: str = targets[batch_i]['annotation']['object'][i]['name']
                target_label: int = label2Idx[target_label_str]
                
                #if len(target_label.shape) == 1:
                #    for ci in range(target_label.shape[0]):
                #        if target_label[ci] > 0:
                #            gts_i.append(torch.cat((target_box, torch.tensor([ci, 0, 0])), dim=-1))
                if target_label > 0:
                    gts_i.append(torch.cat((torch.tensor(target_box), torch.tensor([target_label, 0, 0])), dim=-1))
            gts.append(torch.stack(gts_i) if len(gts_i) > 0 else torch.zeros(0,7))

        #return preds, gts, model

    def map_score(self, preds, gts, model):

        num_classes = 13
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)
        
        assert(len(preds)==len(gts))
        for i in range(len(preds)):
            metric_fn.add(preds[i], gts[i])
        
        metrics = metric_fn.value(iou_thresholds=0.5)

        print(metrics)

        #if model.hparams.test_weights is None:
        #    mapscore = metrics['mAP']
        #else:
        #    weights = np.array(model.hparams.test_weights)        
        #    aps = np.array([metrics[0.5][c]['ap'] for c in metrics[0.5]])
        #
        #    mapscore = (aps * weights).sum()

        with open('./vins_test_pred/metrics.json','w') as f:
            json.dump({metrics}, f)


class _Obsolete(object):
    
    def draw_prediction(pred, test_image):
        draw = ImageDraw.Draw(test_image)

        class_map_file = "../../metadata/screenrecognition/class_map_vins_manual.json"
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        idx2Label = class_map['idx2Label']

        conf_thresh = 0.5
        for i in range(len(pred[0]['boxes'])):
            conf_score = pred[0]['scores'][i]
            if conf_score > conf_thresh:
                x1, y1, x2, y2 = pred[0]['boxes'][i]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                draw.rectangle([x1, y1, x2, y2], outline='red')
                draw.text((x1, y1), idx2Label[str(int(pred[0]['labels'][i]))] + " {:.2f}".format(float(conf_score)), fill="red")

        test_image.show()

    def vins_test_step():
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        
        data = VINSUIDataModule()

        model = UIElementDetector.load_from_checkpoint('../../downloads/checkpoints/screenrecognition-web350k-vins.ckpt')
        test_dataloader = data.test_dataloader()

        ARTIFACT_DIR = "./checkpoints_screenrecognition_web350k-vins"
        logger = TensorBoardLogger(ARTIFACT_DIR)
        
        CHECK_INTERVAL_STEPS = 4000
        #checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, every_n_train_steps=CHECK_INTERVAL_STEPS, save_last=True)
        checkpoint_callback = ModelCheckpoint(dirpath=ARTIFACT_DIR, filename= "screenrecognition",monitor='mAP', mode="max", save_top_k=1)
        
        #earlystopping_callback = EarlyStopping(monitor="mAP", mode="max", patience=10)

        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            callbacks=[checkpoint_callback],
            min_epochs=10,
            logger=logger
        )

        print('Trainer Test Step.......')
        trainer.test(model, data)


if __name__ == "__main__":

    #preds, gts, model = test_pred_gt()
    #map_score(preds, gts, model) # get map score in json file
    '''
    stop = False
    i = 0
    while not stop:
        b = Benchmark(class_map_path="../../metadata/screenrecognition/class_map_vins_manual.json",
                  test_images_path="./vins_test_pred/vins_test_id.json",
                  test_targets_path="./vins_test_pred/vins_test_gt.json",
                  )
        stop = b.getPredGts(iter=i)
        i += 1
    '''

    m = Benchmark(class_map_path="../../metadata/screenrecognition/class_map_vins_manual.json",
                  test_images_path="./vins_test_pred/vins_test_id.json",
                  test_targets_path="./vins_test_pred/vins_test_gt.json",
                  )
    metrics = m.compute_mAP_score(map_type='Pascal-VOC', std=False)
    metrics = m.compute_mAP_score(map_type='COCO', std=False)

    #b.compute_mAP_score()

# VINS Dataset initialise
# get test dataloader
# init Trainer obj
# run Trainer.test -> _test_impl
# hope it works