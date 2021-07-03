import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

torch_loss_dict = torch.nn.modules.loss.__dict__


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class FocalAndCrossEntropyLoss(nn.Module):
    def __init__(self, avg_weight: float = 0.2):
        super(FocalAndCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.avg_weight = avg_weight

    def forward(self, input_tensor, target_tensor):
        return (
            self.avg_weight * self.focal_loss(input_tensor, target_tensor)
            + (1 - self.avg_weight) * self.ce_loss(input_tensor, target_tensor)
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon


    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        return 1 - f1.mean()


# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        # sync device between input and module
        self.fc = self.fc.to(x.device)

        W = self.fc._parameters['weight']
        W = nn.Parameter(F.normalize(W, p=2, dim=1))
        x = F.normalize(x, p=2, dim=1)
        # x = self.fc(x)
    
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(x.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((x[i, :y], x[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        
        return -torch.mean(L)


class CrossEntropyLossAfterSoftmax(nn.Module):
    def __init__(self):
        super(CrossEntropyLossAfterSoftmax, self).__init__()

        
    def forward(self, predictions, targets):
        assert len(predictions) == len(targets)
        
        num_classes = predictions.shape[1]
        
        log_predictions = torch.log(predictions)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).to(log_predictions.dtype)
        
        log_comparisons = torch.sum(-log_predictions * one_hot_targets, dim=1)
        
        return torch.mean(log_comparisons, dim=0)


# class SoftKLDivLoss(nn.Module):
#     def __init__(self, T: float = 1.0):
#         super(SoftKLDivLoss, self).__init__()
#         self.T = T

#     def forward(self, predictions, targets):
#         predictions = F.softmax(predictions / self.T, dim=1)
#         targets = F.softmax(targets / self.T, dim=1)
#         return (targets * torch.log(targets / predictions)).sum().mean()


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, T: float = 1.0):
        super(SoftCrossEntropyLoss, self).__init__()
        self.T = T

    def forward(self, predictions, targets):
        predictions = F.softmax(predictions / self.T, dim=1)
        targets = F.softmax(targets / self.T, dim=1)
        return -(targets * torch.log(predictions)).sum().mean()


class KDLoss(nn.Module):
    def __init__(self, T: float = 1.0, alpha: float = 1.0, num_classes: int = 18):
        super(KDLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, student_outputs, teacher_outputs, targets):
        student_outputs_soft = F.softmax(student_outputs / self.T, dim=1)
        student_outputs_hard = F.softmax(student_outputs, dim=1)
        
        teacher_outputs_soft = F.softmax(teacher_outputs / self.T, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(torch.float)
        
        targets_hard = F.softmax(targets_one_hot, dim=1)

        loss_student = (-targets_hard * torch.log(student_outputs_hard)).sum()
        loss_distill = (-teacher_outputs_soft * torch.log(student_outputs_soft)).sum()

        L = self.alpha * loss_student + (1 - self.alpha) * loss_distill

        return L
