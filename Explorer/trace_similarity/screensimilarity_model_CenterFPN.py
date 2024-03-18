import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models.resnet import Bottleneck

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
        inputs :
          x : input feature maps( B X C X W X H)
        returns :
          out : self attention value + input feature 
          attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out#,attention

class UIScreenEmbedder(pl.LightningModule):
    def __init__(self, lr=0.00005, margin_pos=0.2, margin_neg=0.5, lambda_ocr=0):
        super(UIScreenEmbedder, self).__init__()
        
        self.save_hyperparameters('lr','margin_pos','margin_neg','lambda_ocr')
        
        self.linear5 = nn.Linear(5,1)
        self.linear2 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(256,1,kernel_size=(1,1))
        self.IN = nn.InstanceNorm3d(5) # Instance Norm
        self.self_attn = Self_Attn(256)
        
    def forward_ocr(self, x):
        # handle OCR -- networks to measure text diff in two pages? (given x = ocr of each page separately)
        x = torch.flatten(x, 1)
        return x

    def forward_arch5(self, x):
        #print('FORWARD SHAPE: ', x.shape)

        # x input has shape (N, 2, 5, 256, 40, 44)
        #x_norm_fpn = self.IN(x[:,0,:,:,:,:])
        #x_norm_center = self.IN(x[:,1,:,:,:,:])
        #x = torch.cat((x_norm_fpn.unsqueeze(1), x_norm_center.unsqueeze(1)), dim=1)

        x = torch.permute(x, (0,2,3,4,5,1)) # shape (N,2,5,256,40,44) --> (N,5,256,40,44,2)
        x = self.linear2(x).squeeze(-1) # squeeze: shape (N,5,256,40,44,1) --> (N,5,256,40,44)
        x = self.relu(x)

        x3 = self.self_attn(x[:,0,:,:,:]).unsqueeze(1)
        x4 = self.self_attn(x[:,1,:,:,:]).unsqueeze(1)
        x5 = self.self_attn(x[:,2,:,:,:]).unsqueeze(1)
        x6 = self.self_attn(x[:,3,:,:,:]).unsqueeze(1)
        x7 = self.self_attn(x[:,4,:,:,:]).unsqueeze(1) # each has shape (N,1,256,40,44)
        x = torch.cat((x3,x4,x5,x6,x7),dim=1) # shape (N,5,256,40,44)

        x = torch.permute(x, (0,2,3,4,1))
        x = self.linear5(x)
        x = x.squeeze(-1) # shape (N,256,40,44,1) --> (N,256,40,44)
        x = self.relu(x)

        x = self.conv(x) # shape (N,256,40,44) --> (N,1,40,44)
        
        x = torch.flatten(x, 1) # shape (N,1760) final embedding

        #print('\nFORWARD... x shape: ', x.shape)

        return x

    def forward_p3center3_arch5(self, x):
        x = x[:,:,0,:,:,:] # shape (N,2,265,40,44)
        x = torch.permute(x, (0,2,3,4,1))
        x = self.linear2(x).squeeze(-1) # shape (N,256,40,44)
        x = self.relu(x)
        x = self.self_attn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self.forward_arch5(x) #self.forward_p3center3_arch5(x)

    def get_inputs_from_batch(self, batch):
        fpn1 = batch['fpn1']
        fpn2 = batch['fpn2']
        center1 = batch['center1']
        center2 = batch['center2']
        labels = batch['label']

        batch_size = fpn1.shape[0]
        center1 = torch.broadcast_to(center1, (batch_size, 5, 256, 40, 44))
        center2 = torch.broadcast_to(center2, (batch_size, 5, 256, 40, 44))

        #print('TRAIN SHAPE fpn1: ', fpn1.shape)
        #print('TRAIN SHAPE center1: ', center1.shape)

        in1 = torch.cat((fpn1.unsqueeze(1), center1.unsqueeze(1)), dim=1)
        in2 = torch.cat((fpn2.unsqueeze(1), center2.unsqueeze(1)), dim=1)

        #print('TRAIN SHAPE in1: ', in1.shape)

        return in1, in2, batch_size, labels
    
    def training_step(self, batch, batch_idx):
        in1, in2, batch_size, labels = self.get_inputs_from_batch(batch)

        outs1 = self.forward(in1)
        outs2 = self.forward(in2)

        delta = torch.abs(outs1 - outs2)
        dist = torch.linalg.norm(delta, dim=-1)

        print(f'\ndist of shape {dist.shape}.......: ', dist)
        print(f'\nlabels of shape {labels.shape}.........: ', labels)
        
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
        #fpn1 = batch['fpn1']
        #fpn2 = batch['fpn2']
        #labels = batch['label']

        in1, in2, batch_size, labels = self.get_inputs_from_batch(batch)

        outs1 = self.forward(in1)
        outs2 = self.forward(in2)
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
        #fpn1 = batch['fpn1']
        #fpn2 = batch['fpn2']
        #labels = batch['label']

        in1, in2, batch_size, labels = self.get_inputs_from_batch(batch)

        outs1 = self.forward(in1)
        outs2 = self.forward(in2)
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
