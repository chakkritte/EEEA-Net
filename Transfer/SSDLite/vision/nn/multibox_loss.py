import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable



from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

        self.focalloss = FocalLoss(21, gamma=2, size_average = False)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')
        #classification_loss = self.focalloss(confidence.view(-1, num_classes), labels)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        #smooth_l1_loss = ciou_loss(predicted_locations, gt_locations, eps=1e-7, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        print(self.gamma)
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim= 1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def ciou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag

    # calculate v,alpha
    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou
    if reduction == 'mean':
        loss = torch.mean(ciou_loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    return loss