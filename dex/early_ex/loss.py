import pytorch_metric_learning
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.distances import LpDistance
import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
import pytorch_metric_learning.utils.common_functions as cf
import torch
import torch.nn.functional as F
from collections import defaultdict
from early_ex.utils import *
class ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
def concat_indices_tuple(x):
  return [torch.cat(y) for y in zip(*x)]

class BoneCenterLoss(BaseMetricLossFunction):
  def __init__(
      self,
      margin=0.5,
      swap=False,
      smooth_loss=False,
      triplets_per_anchor="all",
      **kwargs
  ):
    super().__init__(**kwargs)
    self.triplet_loss = pytorch_metric_learning.losses.TripletMarginLoss(
        margin=margin,
        swap=swap,
        smooth_loss=smooth_loss,
        triplets_per_anchor=triplets_per_anchor,
        **kwargs
    )
  def create_masks_train(self, class_labels):
    labels_dict = defaultdict(list)
    class_labels = class_labels.detach().cpu().numpy()
    for idx, pid in enumerate(class_labels):
      labels_dict[pid].append(idx)

    unique_classes = list(labels_dict.keys())
    labels_list = list(labels_dict.values())
    lens_list = [len(item) for item in labels_list]
    lens_list_cs = np.cumsum(lens_list)

    M = max(len(instances) for instances in labels_list)
    P = len(unique_classes)

    query_indices = []
    class_masks = torch.zeros((P, len(class_labels)), dtype=bool)
    masks = torch.zeros((M * P, len(class_labels)), dtype=bool)
    for class_idx, class_insts in enumerate(labels_list):
      class_masks[class_idx, class_insts] = 1
      for instance_idx in range(M):
        matrix_idx = class_idx * M + instance_idx
        if instance_idx < len(class_insts):
          query_indices.append(class_insts[instance_idx])
          ones = class_insts[:instance_idx] + class_insts[instance_idx + 1 :]
          masks[matrix_idx, ones] = 1
        else:
          query_indices.append(0)
      return masks, class_masks, labels_list, query_indices

  def forward(
    self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        self.reset_stats()
        cf.check_shapes(embeddings, labels)
        labels = cf.to_device(labels, embeddings)
        loss_dict = self.compute_loss(
            embeddings, labels, 
            indices_tuple=indices_tuple, 
            ref_emb=ref_emb, ref_labels=ref_labels )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

  def compute_loss(self, embeddings, labels, indices_tuple, margin=0.05, ref_emb=None, ref_labels=None):
    # print(embeddings.shape)
    # print(labels.shape)

    masks, class_masks, labels_list, query_indices = self.create_masks_train(labels)
    # # P = len(labels_list)
    # # M = max([len(instances) for instances in labels_list])
    # # DIM = embeddings.size(-1)
    # # print(masks.shape)
    # # print(class_masks.shape)
    # # print(labels_list)
    # masks_float = masks.type(embeddings.type()).to(embeddings.device)
    # class_masks_float = class_masks.type(embeddings.type()).to(embeddings.device)
    # inst_counts = masks_float.sum(-1)
    # class_inst_counts = class_masks_float.sum(-1)
    # print('inst_counts:',inst_counts)
    # print('class_inst_counts:',class_inst_counts)
    uni_labels = torch.unique(labels)

    # padded = masks_float.unsqueeze(-1) * embeddings.unsqueeze(0)
    # class_padded = class_masks_float.unsqueeze(-1) * embeddings.unsqueeze(0)

    # print(padded)
    # print(class_padded)

    # # print(uni_labels)
    # num_class = len(uni_labels)
    class_emb = torch.zeros((len(labels_list), embeddings.shape[-1])).to(embeddings.device)
    # # print(class_emb.shape)
    for y in uni_labels:
    #   # print(y)
      class_emb[y] = torch.mean(embeddings[labels_list[y]],dim=0)

    F.normalize(class_emb,dim=1)
    
    loss = self.triplet_loss.compute_loss(
      embeddings, labels, indices_tuple=None, ref_emb=class_emb, ref_labels=uni_labels)
    return loss
    

    # class_emb = F.normalize(class_emb, dim=1)
    # n, d  = embeddings.size(0), embeddings.size(1)
    # m     = class_emb.size(0)

    # embeddings = embeddings.unsqueeze(1).expand(n, m, d)
    # class_emb = class_emb.unsqueeze(0).expand(n, m, d)
    # # print(embeddings.shape)
    # # print(class_emb.shape)
    # d = torch.nn.PairwiseDistance(p=2)
    # dist = d(embeddings, class_emb)
    # # print(dist.shape)
    # # print(minn)

    


    # dists = dist.amin(1) - dist.amax(1) + margin
    # zeros = torch.zeros_like(dists)
    # losss = torch.maximum(dists, zeros)
    # loss = torch.mean(losss)
    # print(dists.shape)
    # print(loss)
    # dist = d(a, p) - d(a, n) + margin
    # loss = torch.mean(torch.max(dist, torch.zeros_like(dist)))
    # return loss



class BoneLoss(BaseMetricLossFunction):
  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      # self.temperature = torch.autograd.Variable(torch.Tensor([temperature]), requires_grad= True).cuda()
      self.distance = LpDistance(power=2)

  def forward(
      self, embeddings, labels, indices_tuple=None, 
      ref_emb=None, ref_labels=None, temperature=None):

      self.reset_stats()
      cf.check_shapes(embeddings, labels)
      labels = cf.to_device(labels, embeddings)
      ref_emb, ref_labels = cf.set_ref_emb(
        embeddings, labels, ref_emb, ref_labels)
      loss_dict = self.compute_loss(
          embeddings, labels, indices_tuple, ref_emb, ref_labels, temperature)

      self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
      return self.reducer(loss_dict, embeddings, labels)


  def compute_loss(
      self, embeddings, labels, indices_tuple, ref_emb, ref_labels, temperature=None):
      # perform some calculation #

      dtype = embeddings.dtype
      miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
      if temperature != None:
        mat = torch.div( - self.distance(embeddings, ref_emb), temperature)
      else:
        mat = - self.distance(embeddings, ref_emb)
      exp = F.softmax(mat, dim=1)
      same_class = cf.to_dtype(
          labels.unsqueeze(1) == ref_labels.unsqueeze(0), dtype=dtype)
      exp = torch.sum(exp * same_class, dim=1)
      non_zero = exp != 0
      some_loss = -torch.log(exp[non_zero]) * miner_weights[non_zero]

      # put into dictionary #
      return {
          "loss": {
              "losses": some_loss,
              "indices": cf.torch_arange_from_size(embeddings)[non_zero],
              "reduction_type": "element",
          }
      }


import numpy as np
import torch
from numba import jit
from torch.autograd import Function

@jit(nopython = True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  for k in range(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = max(max(r0, r1), r2)
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  E = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in range(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma=gamma
    self.func_dtw = _SoftDTW.apply

  def calc_distance_matrix(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    return dist

  def forward(self, x, y):
    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) < 3:
      x = x.unsqueeze(0)
      y = y.unsqueeze(0)
      squeeze = True
    if self.normalize:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      D_xx = self.calc_distance_matrix(x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      D_yy = self.calc_distance_matrix(y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      result = out_xy - 1/2 * (out_xx + out_yy) # distance
    else:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      result = out_xy # discrepancy
    return result.squeeze(0) if squeeze else result

    
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

class Proxy_Anchor(torch.nn.Module):
  def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
    torch.nn.Module.__init__(self)
    # Proxy Anchor Initialization
    self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
    nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    self.nb_classes = nb_classes
    self.sz_embed = sz_embed
    self.mrg = mrg
    self.alpha = alpha
      
  def forward(self, X, T):
    P = self.proxies

    cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
    P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
    N_one_hot = 1 - P_one_hot

    pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
    neg_exp = torch.exp(self.alpha * (cos + self.mrg))

    with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
    num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
    
    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
    N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
    
    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
    neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
    loss = pos_term + neg_term     
    
    return loss