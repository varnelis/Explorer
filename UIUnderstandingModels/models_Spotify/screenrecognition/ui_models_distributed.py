import numpy as np
import json
# from torchmetrics.detection.mean_ap import MeanAveragePrecision

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# suppress pandas FutureWarning for mean_average_precision

from mean_average_precision import MetricBuilder
import pytorch_lightning as pl
import torch
import torchvision
import random
from tqdm import tqdm
from time import time

from PIL import Image, ImageDraw

from ui_models_extra import FCOSMultiHead

class UIElementDetector(pl.LightningModule):
    def __init__(self, num_classes=25, min_size=320, max_size=640, use_multi_head=True, lr=0.0001, val_weights=None, test_weights=None, arch="fcos"):
        super(UIElementDetector, self).__init__()
        self.save_hyperparameters()
        self.hparams.lr = lr
        self.hparams.val_weights = val_weights
        self.hparams.test_weights = test_weights
        print(f'UIElementDetector -- lr = {self.hparams.lr}')
        if arch == "fcos":
            model = torchvision.models.detection.fcos_resnet50_fpn(min_size=min_size, max_size=max_size, num_classes=num_classes, trainable_backbone_layers=5)
            if use_multi_head:
                multi_head = FCOSMultiHead(model.backbone.out_channels, model.anchor_generator.num_anchors_per_location()[0], num_classes)
                model.head = multi_head
        elif arch == "ssd":
            model = torchvision.models.detection.ssd300_vgg16(num_classes=num_classes, trainable_backbone_layers=5)
        self.model = model
        self.test_step_time = 0

    def training_step(self, batch, batch_idx):
        #print('TRAINING STEP')
        
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict({
            'train_loss': float(loss)
        }, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #print(f'Validation {batch_idx}')

        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        preds = []
        gts = []
        for batch_i in tqdm(range(len(outputs))):

            #print('VAL batch : ', batch_i)

            batch_len = outputs[batch_i]['boxes'].shape[0]
            # print(outputs[batch_i]['boxes'])
            # print(len(outputs), batch_len)
            pred_box = outputs[batch_i]['boxes']
            pred_score = outputs[batch_i]['scores']
            pred_label = outputs[batch_i]['labels']
            # preds.append({
            #     'boxes': pred_box.to(self.device),
            #     'scores': pred_score.to(self.device),
            #     'labels': pred_label.to(self.device) - 1
            # })
            preds.append(torch.cat((pred_box, pred_label.unsqueeze(-1), pred_score.unsqueeze(-1)), dim=-1))
            
            gtsi = []
            target_len = targets[batch_i]['boxes'].shape[0]
            for i in range(target_len):
                target_box = targets[batch_i]['boxes'][i]
                target_label = targets[batch_i]['labels'][i]
                
                if len(target_label.shape) == 1:
                    for ci in range(target_label.shape[0]):
                        if target_label[ci] > 0:
                            gtsi.append(torch.cat((target_box, torch.tensor([ci, 0, 0], device=target_box.device)), dim=-1))
                else:
                    gtsi.append(torch.cat((target_box, torch.tensor([target_label, 0, 0], device=target_box.device)), dim=-1))
            gts.append(torch.stack(gtsi) if len(gtsi) > 0 else torch.zeros(0, 7, device=self.device))

            #print('VAL batch END : ', batch_i)

        return preds, gts

    def validation_epoch_end(self, outputs):
        #print('VAL EPOCH END.........')

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=self.hparams.num_classes)
        for batch_output in outputs:
            for i in range(len(batch_output[0])):
                metric_fn.add(batch_output[0][i].detach().cpu().numpy(), batch_output[1][i].detach().cpu().numpy())
            
        # print(torch.cat([torch.stack(o[0]) for o in outputs], dim=0).shape, torch.cat([torch.stack(o[0]) for o in outputs], dim=0).sum())
            
        metrics = metric_fn.value(iou_thresholds=0.5)
        print(np.array([metrics[0.5][c]['ap'] for c in metrics[0.5]]))
        
        if self.hparams.val_weights is None:
            mapscore = metrics['mAP']
        else:
            weights = np.array(self.hparams.val_weights)
            # weights = weights[1:]
         
            aps = np.array([metrics[0.5][c]['ap'] for c in metrics[0.5]])  

            mapscore = (aps * weights).sum()                   
        
        self.log_dict({'val_mAP': mapscore}, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        #print('TEST STEP...............')

        test_start = time()

        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        preds = []
        gts = []
        for batch_i in range(len(outputs)):
            batch_len = outputs[batch_i]['boxes'].shape[0]
            # print(outputs[batch_i]['boxes'])
            # print(len(outputs), batch_len)
            pred_box = outputs[batch_i]['boxes']
            pred_score = outputs[batch_i]['scores']
            pred_label = outputs[batch_i]['labels']
            # preds.append({
            #     'boxes': pred_box.to(self.device),
            #     'scores': pred_score.to(self.device),
            #     'labels': pred_label.to(self.device) - 1
            # })
            preds.append(torch.cat((pred_box, pred_label.unsqueeze(-1), pred_score.unsqueeze(-1)), dim=-1))
            
            gtsi = []
            target_len = targets[batch_i]['boxes'].shape[0]
            for i in range(target_len):
                target_box = targets[batch_i]['boxes'][i]
                target_label = targets[batch_i]['labels'][i]
                
                if len(target_label.shape) == 1:
                    for ci in range(target_label.shape[0]):
                        if target_label[ci] > 0:
                            gtsi.append(torch.cat((target_box, torch.tensor([ci, 0, 0], device=target_box.device)), dim=-1))
                else:
                    gtsi.append(torch.cat((target_box, torch.tensor([target_label, 0, 0], device=target_box.device)), dim=-1))
            gts.append(torch.stack(gtsi) if len(gtsi) > 0 else torch.zeros(0, 7, device=self.device))

        test_time = time() - test_start
        self.test_step_time += test_time
        self.log_dict({"test_time": self.test_step_time}, sync_dist=True)

        return preds, gts, images
    
    def test_epoch_end(self, outputs):
        #print('TEST EPOCH END...............')
        
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=self.hparams.num_classes)

        mAP_per_test_item = {}
        #with open('tensor_to_img.json','r') as f:
        #    t2img = json.load(f)
        #ti = {int(k):v for k,v in t2img.items()}

        idx_count = 0
        for batch_output in outputs:
            for i in range(len(batch_output[0])):
                metric_fn.add(batch_output[0][i].detach().cpu().numpy(), batch_output[1][i].detach().cpu().numpy())

                # get specific item mAP & log
                metric_fn_i = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=self.hparams.num_classes)
                metric_fn_i.add(batch_output[0][i].detach().cpu().numpy(), batch_output[1][i].detach().cpu().numpy())
                mAP_i = self._metric_calc(metric_fn_i, 'test_items_i_mAP')
                
                #uuid_i = ti[idx_count]

                mAP_per_test_item[idx_count] = mAP_i

                # draw each test image with the pred and true boxes
                bbox_preds_all = batch_output[0][i].detach().cpu().tolist()
                bbox_preds = []
                for bbox in bbox_preds_all:
                    if bbox[-1] >= 0.5:
                        bbox_preds.append(bbox)
                bbox_true = batch_output[1][i].detach().cpu()
                drawn_bbox = self._draw_bbox(bbox_preds, bbox_true, batch_output[2][i].detach())
                
                if drawn_bbox == 0:
                    print(f'\ninvalid img input at idx {idx_count}... skip\n')
                else:
                    drawn_bbox.save(f'./test_img_bbox/img_{idx_count}.png')

                idx_count += 1

        print("=======================================")
        print(f"Test mAP per item i: {mAP_per_test_item}")
        print("=======================================\n")

        #with open('map_img.json', 'w') as f:
        #    json.dump(mAP_per_test_item, f)

        self._metric_calc(metric_fn, 'test_mAP')

    def _metric_calc(self, metric_fn, log_key):
        metrics = metric_fn.value(iou_thresholds=0.5)
        
        print(np.array([metrics[0.5][c]['ap'] for c in metrics[0.5]]))
        
        if self.hparams.test_weights is None:
            mapscore = metrics['mAP']
        else:
            weights = np.array(self.hparams.test_weights)
            # weights = weights[1:]
         
            aps = np.array([metrics[0.5][c]['ap'] for c in metrics[0.5]])  

            mapscore = (aps * weights).sum()                   
        
        self.log_dict({log_key: mapscore}, sync_dist=True)

        return mapscore

    def _draw_bbox(self, bbox_preds, bbox_true, img, scale_factor = 1):
        if img is None:
            return 0

        if len(bbox_preds) == 0:
            print('no boxes at confidence >= 0.5 here')
    
        img_transform = torchvision.transforms.functional.to_pil_image(img)
    
        draw_bbox = ImageDraw.Draw(img_transform)
        for i, bbox in enumerate(bbox_preds):
            scaled_bbox = [e * scale_factor for e in bbox[:4]]
            draw_bbox.rectangle(scaled_bbox, outline = "red", width = 3)
    
        for i, bbox in enumerate(bbox_true):
            scaled_bbox = [e * scale_factor for e in bbox[:4]]
            draw_bbox.rectangle(scaled_bbox, outline = "green", width = 3)

        return img_transform

    def configure_optimizers(self):
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
