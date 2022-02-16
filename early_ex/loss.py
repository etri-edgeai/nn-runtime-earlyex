from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.distances import LpDistance
import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
import pytorch_metric_learning.utils.common_functions as cf
import torch
import torch.nn.functional as F

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

class BoneLoss(BaseMetricLossFunction):
    def __init__(self, temperature=1, **kwargs):
        super().__init__(**kwargs)
        # self.temperature = torch.autograd.Variable(torch.Tensor([temperature]), requires_grad= True).cuda()
        self.distance = LpDistance(power=2)

    def forward(
        self, embeddings, labels, indices_tuple=None, 
        ref_emb=None, ref_labels=None, temperature=None):

        self.reset_stats()
        cf.check_shapes(embeddings, labels)
        labels = cf.to_device(labels, embeddings)
        ref_emb, ref_labels = cf.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels, temperature
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)


    def compute_loss(
        self, embeddings, labels, indices_tuple, ref_emb, ref_labels, temperature=None):
        # perform some calculation #

        dtype = embeddings.dtype
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mat = torch.div( - self.distance(embeddings, ref_emb), temperature)
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