import torch
from torch import nn
import numpy as np
import glob
import cv2
import os


class LossCombiner(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.device = device

        for k, v in kwargs.items():
            self.losses[k] = v[0]
            self.weights[k] = v[1]

    def update_weights(self, new_weights):
        self.weights = new_weights

    def forward(self, loss_dict, return_individuals=False):
        loss = torch.tensor(0).to(self.device)
        loss_values = {}
        for k, v in loss_dict.items():
            loss_value = self.losses[k](*v)
            loss_values[k] = loss_value
            loss = loss + self.weights[k] * loss_value

        if (return_individuals):
            return loss, loss_values
        else:
            return loss


class DoubleLoss(nn.Module):
    def __init__(self, l1_weighting_factor=1.0, reduction = 'mean'):
        super().__init__()
        self.MSE = torch.nn.MSELoss(reduction = reduction)
        self.L1 = torch.nn.L1Loss(reduction = reduction)
        self.l1_weighting_factor = l1_weighting_factor

    def forward(self, predictions, labels, segmentation_mask=None):

        if (segmentation_mask is not None):
            return self.MSE(segmentation_mask * predictions, labels) + self.l1_weighting_factor * self.L1(
                segmentation_mask * predictions, labels)
        else:
            return self.MSE(predictions, labels) + self.l1_weighting_factor * self.L1(predictions, labels)




class ScaleLoss(nn.Module):
    def __init__(self, normalised_1, factor = 1.0):
        super().__init__()
        self.MSE = torch.nn.MSELoss(reduction = 'none')
        self.normalised_1 = normalised_1
        self.factor = factor

    def forward(self, predictions, labels):
        # find which labels have scale below1
        rescaling_array = (labels < self.normalised_1).float()*self.factor

        mse_unreduced = self.MSE(predictions, labels)

        rescaling = torch.ones_like(mse_unreduced)
        rescaling = rescaling + rescaling_array

        mse_unreduced = mse_unreduced*rescaling

        return torch.mean(mse_unreduced)







class ContrastiveDonLoss(nn.Module):
    def __init__(self,
                 distance_metric = 'mse',
                 margin_bg = 10.,
                 margin_obj = 10.,
                 hard_negative_scaling = False):
        super().__init__()

        self.positive_loss = self.select_loss(distance_metric, reduction = 'mean' )
        self.negative_loss = self.select_loss(distance_metric, reduction = 'none' )

        self.margin_bg = margin_bg
        self.margin_obj = margin_obj

        self.hard_negative_scaling = hard_negative_scaling


    def select_loss(self,distance_metric , reduction = 'mean'):
        if(distance_metric == 'mse'):
            return torch.nn.MSELoss(reduction = reduction)
        elif(distance_metric == 'l1'):
            return torch.nn.L1Loss(reduction = reduction)
        elif(distance_metric == 'double'):
            return DoubleLoss(reduction=reduction)
        else:
            raise NotImplementedError()


    def forward(self,outs_from_correct, outs_to_correct, outs_from_obj_wrong, outs_to_obj_wrong , outs_from_wrong=None, outs_to_wrong=None ):
        loss_correct = self.positive_loss(outs_from_correct, outs_to_correct)

        if(outs_from_wrong is not None):
            # outs_from_wrong is N*C*Samples -> N*Samples
            loss_wrong =  self.margin_bg - self.negative_loss(outs_from_wrong, outs_to_wrong).mean(1)
            loss_wrong = torch.clamp(loss_wrong, min=0.)

            if(self.hard_negative_scaling):
                scale_factor = (loss_wrong != 0).detach().sum()
                loss_wrong = loss_wrong.sum() / torch.maximum(scale_factor.float(), torch.tensor(1e-6).float())
            else:
                loss_wrong = loss_wrong.mean()
        else:
            loss_wrong = torch.tensor(0.)

        # outs_from_obj_wrong is N*C*Samples -> N*Samples
        loss_o_wrong = self.margin_bg - self.negative_loss(outs_from_obj_wrong, outs_to_obj_wrong).mean(1)
        loss_o_wrong = torch.clamp(loss_o_wrong, min=0.)

        if (self.hard_negative_scaling):
            scale_factor = (loss_o_wrong != 0).detach().sum()
            loss_o_wrong = loss_o_wrong.sum() / torch.maximum(scale_factor.float(), torch.tensor(1e-6).float())
        else:
            loss_o_wrong = loss_o_wrong.mean()

        loss = loss_correct + loss_o_wrong + loss_wrong

        return loss, loss_correct.item(), loss_o_wrong.item(), loss_wrong.item()



def calculate_focal_loss_alpha(path_to_masks=None,  mask_files = None):

    if(mask_files is None):
        mask_files_png = glob.glob(os.path.join(path_to_masks, '*.png'))
        mask_files_jpg = glob.glob(os.path.join(path_to_masks, '*.jpg'))
        mask_files = mask_files_jpg+mask_files_png

    total_positive = 0

    for i, mask_file in enumerate(mask_files):
        mask = cv2.imread(mask_file, -1)
        if i == 0:
            height, width = mask.shape

        total_positive += np.sum(mask)

    total_num_pixels = height * width * len(mask_files)
    total_negative = total_num_pixels - total_positive
    alpha = total_negative / total_num_pixels
    return alpha

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, epsilon=1e-8, logits=True, device=torch.device('cuda:0')):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = torch.tensor([epsilon]).to(device)
        self.logits = logits
        if self.logits:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred, mask):
        if self.logits:
            pred = self.sigmoid(pred)

        loss = - mask * self.alpha * (1 - pred) ** self.gamma * torch.log(torch.max(pred, self.epsilon)) \
               - (1 - mask) * (1 - self.alpha) * (pred) ** self.gamma * torch.log(torch.max(1 - pred, self.epsilon))

        return loss.mean()




class ContrastiveDonLossWithSeg(nn.Module):
    def __init__(self,
                 distance_metric = 'mse',
                 margin_bg=10.,
                 margin_obj=10.,
                 hard_negative_scaling=False,
                 alpha=0.5,
                 gamma=2,
                 epsilon=1e-8,
                 logits=True,
                 device=torch.device('cuda:0'),
                 seg_relative_weight = 1.0):
        super().__init__()

        self.contrastive_l = ContrastiveDonLoss(
            distance_metric,
            margin_bg,
            margin_obj,
            hard_negative_scaling)
        self.focal_l = FocalLoss(
            alpha,
            gamma,
            epsilon,
            logits,
            device
        )

        self.seg_relative_weight = seg_relative_weight


    def forward(self,outs_from_correct, outs_to_correct,outs_from_obj_wrong, outs_to_obj_wrong, outs_from_wrong, outs_to_wrong,  from_seg_out, from_seg_gt, to_seg_out, to_seg_gt):

        don_loss, don_loss_correct, don_loss_o_wrong, don_loss_wrong = self.contrastive_l(outs_from_correct, outs_to_correct, outs_from_obj_wrong, outs_to_obj_wrong, outs_from_wrong, outs_to_wrong, )
        from_seg_loss = self.focal_l(from_seg_out, from_seg_gt)
        to_seg_loss = self.focal_l(to_seg_out, to_seg_gt)
        seg_loss = (from_seg_loss+to_seg_loss)/2.

        loss = don_loss + self.seg_relative_weight *  seg_loss


        return  loss, don_loss, seg_loss,  don_loss_correct, don_loss_o_wrong, don_loss_wrong





