import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models.resnet import Bottleneck

class UIScreenEmbedder(pl.LightningModule):
    def __init__(self, hidden_size=256, lr=0.00005, margin_pos=0.2, margin_neg=0.5, lambda_ocr=0):
        super(UIScreenEmbedder, self).__init__()
        
        self.save_hyperparameters('lr','margin_pos','margin_neg','lambda_ocr')
        
        self.linear = nn.Linear(5,1)
        self.relu = nn.ReLU()
        self.IN = nn.InstanceNorm2d(hidden_size) # Instance Norm
        self.bottleneck = Bottleneck(hidden_size,64,base_width=4,groups=32)
        
    def forward_ocr(self, x):
        # handle OCR -- networks to measure text diff in two pages? (given x = ocr of each page separately)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # x input has shape (N, 5, 256, 40, 44)
        x = torch.permute(x, (0,2,3,4,1))
        x = self.linear(x)
        x = x.squeeze(-1) # squeeze-out the  linear-1 dim: shape (N,256,40,44,1) --> (N,256,40,44)
        #x = self.relu(x)

        x = self.IN(x) # Instance Norm
        x = self.bottleneck(x) # ResNeXt block (with Batch Norm)
        
        x = torch.flatten(x, 1)

        return x
    
    def training_step(self, batch, batch_idx):
        fpn1 = batch['fpn1']
        fpn2 = batch['fpn2']
        labels = batch['label']

        outs1 = self.forward(fpn1)
        outs2 = self.forward(fpn2)
        batch_size = fpn1.shape[0]
        delta = outs1 - outs2
        dist = torch.linalg.norm(delta, dim=-1)
        
        losses = torch.zeros(batch_size, device=self.device)
        losses[labels] = (dist[labels] - self.hparams.margin_pos).clamp(min=0)
        losses[~labels] = (self.hparams.margin_neg - dist[~labels]).clamp(min=0)
        loss_sim = losses.mean()
        
        if self.hparams.lambda_ocr == 0:
            loss = loss_sim
            metrics = {'train_loss': loss}
            self.log_dict(metrics)
            return loss
        else:
            # separate forward func for OCR and add OCR dist to the screensim loss
            '''
            ocr_pred_outs1 = self.forward_uda(image1)
            ocr_pred_outs2 = self.forward_uda(image2)
            cls_pred_outsuda1 = self.forward_uda(imageuda1)
            cls_pred_outsuda2 = self.forward_uda(imageuda2)
            cls_pred = torch.cat((cls_pred_outs1, cls_pred_outs2, cls_pred_outsuda1, cls_pred_outsuda2), dim=0).squeeze(-1)
            cls_label = torch.cat((torch.ones(batch_size * 2, device=self.device), torch.zeros(batch_size * 2, device=self.device)), dim=0)
            loss_cls = F.binary_cross_entropy_with_logits(cls_pred, cls_label)
            loss = loss_sim + self.hparams.lambda_dann * loss_cls
            metrics = {'loss': loss, 'loss_sim': loss_sim, 'loss_cls': loss_cls}
            self.log_dict(metrics)
            return loss
            '''
            loss = loss_sim
            metrics = {'train_loss': loss}
            self.log_dict(metrics, sync_dist=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        fpn1 = batch['fpn1']
        fpn2 = batch['fpn2']
        labels = batch['label']

        outs1 = self.forward(fpn1)
        outs2 = self.forward(fpn2)
        delta = outs1 - outs2
        dist = torch.linalg.norm(delta, dim=-1)

        thresh = 0.5 * (self.hparams.margin_pos + self.hparams.margin_neg)
        preds = dist < thresh

        if self.hparams.lambda_ocr == 0:
            return preds, labels
        else:
            '''
            cls_pred_outs1 = self.forward_uda(image1)
            cls_pred_outs2 = self.forward_uda(image2)
            cls_pred_outsuda1 = self.forward_uda(imageuda1)
            cls_pred_outsuda2 = self.forward_uda(imageuda2)
            cls_pred = torch.cat((cls_pred_outs1, cls_pred_outs2, cls_pred_outsuda1, cls_pred_outsuda2), dim=0).squeeze(-1) > 0
            cls_label = torch.cat((torch.ones(batch_size * 2, device=self.device), torch.zeros(batch_size * 2, device=self.device)), dim=0)
            return preds, labels, cls_pred, cls_label
            '''
            return preds, labels
    
    def validation_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        score = f1_score(all_labels.detach().cpu().numpy(), all_outs.detach().cpu().numpy())
        
        print('Val F1: ', score)

        if self.hparams.lambda_ocr == 0:
            metrics = {'val_f1': score}
            self.log_dict(metrics, sync_dist=True)
        else:
            '''
            all_outs_uda = torch.cat([o[2] for o in outputs], dim=0)
            all_labels_uda = torch.cat([o[3] for o in outputs], dim=0)
            score_uda = f1_score(all_labels_uda.detach().cpu().numpy(), all_outs_uda.detach().cpu().numpy())
            
            metrics = {'f1': score, 'f1_uda': score_uda}
            self.log_dict(metrics)
            '''
            metrics = {'val_f1': score}
            self.log_dict(metrics, sync_dist=True)

    def test_step(self, batch, batch_idx):
        fpn1 = batch['fpn1']
        fpn2 = batch['fpn2']
        labels = batch['label']

        outs1 = self.forward(fpn1)
        outs2 = self.forward(fpn2)
        delta = outs1 - outs2
        dist = torch.linalg.norm(delta, dim=-1)

        thresh = 0.5 * (self.hparams.margin_pos + self.hparams.margin_neg)
        preds = dist < thresh

        if self.hparams.lambda_ocr == 0:
            return preds, labels
        else:
            '''
            cls_pred_outs1 = self.forward_uda(image1)
            cls_pred_outs2 = self.forward_uda(image2)
            cls_pred_outsuda1 = self.forward_uda(imageuda1)
            cls_pred_outsuda2 = self.forward_uda(imageuda2)
            cls_pred = torch.cat((cls_pred_outs1, cls_pred_outs2, cls_pred_outsuda1, cls_pred_outsuda2), dim=0).squeeze(-1) > 0
            cls_label = torch.cat((torch.ones(batch_size * 2, device=self.device), torch.zeros(batch_size * 2, device=self.device)), dim=0)
            return preds, labels, cls_pred, cls_label
            '''
            return preds, labels
    
    def test_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        score = f1_score(all_labels.detach().cpu().numpy(), all_outs.detach().cpu().numpy())
        
        if self.hparams.lambda_ocr == 0:
            metrics = {'test_f1': score}
            self.log_dict(metrics, sync_dist=True)
        else:
            '''
            all_outs_uda = torch.cat([o[2] for o in outputs], dim=0)
            all_labels_uda = torch.cat([o[3] for o in outputs], dim=0)
            score_uda = f1_score(all_labels_uda.detach().cpu().numpy(), all_outs_uda.detach().cpu().numpy())
            
            metrics = {'f1': score, 'f1_uda': score_uda}
            self.log_dict(metrics)
            '''
            metrics = {'test_f1': score}
            self.log_dict(metrics, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)
        return optimizer
