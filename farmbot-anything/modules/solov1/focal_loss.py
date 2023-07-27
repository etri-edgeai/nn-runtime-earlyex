
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, targets):
        # inputs = torch.sigmoid(inputs)
        # if self.with_logits:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # else:
        # BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        # if self.reduction == 'mean':
        return torch.mean(F_loss)
        # elif self.reduction == 'sum':
        # return torch.sum(F_loss)
        # elif self.reduction == 'none':
        # return F_loss
        # else:
        #     raise ValueError('Unsupported reduction mode.')


# class FocalLoss(nn.Module):

#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
   
#     def forward(self, inputs, targets):
#         sigmoid_input = torch.sigmoid(inputs)
#         term1 = (1 - sigmoid_input) ** self.gamma * targets * torch.log(sigmoid_input.clamp(min=1e-5))
#         term2 = sigmoid_input ** self.gamma * (1 - targets) * torch.log((1 - sigmoid_input).clamp(min=1e-5))
#         loss = -self.alpha * (term1 + term2)

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise ValueError('Unsupported reduction mode.')

# def sigmoid_focal_loss(logits, targets, gamma, alpha):
#     num_classes = logits.shape[1]
#     dtype = targets.dtype
#     device = targets.device
#     class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

#     t = targets.unsqueeze(1)
#     p = torch.sigmoid(logits)
#     term1 = (1 - p) ** gamma * torch.log(p)
#     term2 = p ** gamma * torch.log(1 - p)
#     return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)

# class FocalLoss(nn.Module):
#     def __init__(self,
#                  gamma=2.0,
#                  alpha=0.25,
#                  reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
 
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
# #        print(weight,self.gamma,self.alpha,reduction,avg_factor)
#         loss_cls = sigmoid_focal_loss(pred, target, self.gamma, self.alpha)
#         return loss_cls