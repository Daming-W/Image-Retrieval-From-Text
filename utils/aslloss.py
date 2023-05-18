"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        # _loss = - self.loss.sum()
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss

class KDLoss(nn.Module):
    """Knowledge distillation loss (KL divergence)."""

    name = 'kd_loss'

    def __init__(self, temperature, eps=1e-9):
        super(KDLoss, self).__init__()
        self.T = temperature
        self.eps = eps

    def _to_log_distrib(self, outs: torch.Tensor) -> torch.Tensor:
        outs_sigmoid = torch.sigmoid(outs.reshape(-1) / self.T)
        return torch.log(torch.stack((outs_sigmoid, 1-outs_sigmoid), dim=1)+self.eps)

    def forward(self, stu_outs: torch.Tensor, tea_outs: torch.Tensor) -> torch.Tensor:
        kl_div = nn.functional.kl_div(
            self._to_log_distrib(stu_outs),
            self._to_log_distrib(tea_outs),
            reduction="batchmean", log_target=True)
        return self.T ** 2 * kl_div

def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.sqrt(torch.pow(X, 2).sum(dim=dim, keepdim=True) + eps)
    X = torch.div(X, norm)
    return X

class Feacriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,target,weight = None, pad=False):
        x_norm = l2norm(x)
        target_norm = l2norm(target)
        loss = torch.sqrt(torch.pow((x_norm-target_norm),2).sum(dim=-1)).squeeze()
        if weight != None:
            if pad:
                loss = loss * ((1/weight**2)+1e-12) + torch.log(weight**2+1e-12)
            else:
                loss = loss * weight
            loss = loss.sum(dim=-1).mean()
        else:
            while len(loss.size())>1:
                loss = loss.sum(dim=-1)
            loss = loss.mean()
        return loss

class LogLoss(nn.Module):
    """Knowledge distillation loss (KL divergence)."""

    name = 'kd_loss'

    def __init__(self, eps=1e-6):
        super(LogLoss, self).__init__()
        self.eps = eps

    def forward(self, stu_outs: torch.Tensor, tea_outs: torch.Tensor) -> torch.Tensor:
        stu_outs = torch.sigmoid(stu_outs)
        tea_outs = torch.sigmoid(tea_outs)
        loss = -1*tea_outs*torch.log(stu_outs+self.eps)
        loss = loss.sum(dim=-1).mean(dim=0)
        return loss