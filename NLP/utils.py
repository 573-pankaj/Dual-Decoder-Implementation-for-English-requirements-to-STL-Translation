import torch
from d2l import torch as d2l

class MaskedSoftmaxCELossWithLS(d2l.MaskedSoftmaxCELoss):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, pred, label, valid_len):
        # pred: [B, T, V], label: [B, T]
        # Apply label smoothing
        num_classes = pred.size(-1)
        with torch.no_grad():
            smooth_labels = torch.zeros_like(pred).scatter_(
                2, label.unsqueeze(2), 1.0 - self.eps
            )
            smooth_labels += self.eps / num_classes
        # Standard masked CE
        weights = torch.ones_like(label)
        weights = d2l.sequence_mask(weights, valid_len, value=0)
        unweighted_loss = -(smooth_labels * pred.log_softmax(dim=-1)).sum(dim=-1)
        weighted_loss = (unweighted_loss * weights).sum(dim=1) / valid_len
        return weighted_loss.mean()
