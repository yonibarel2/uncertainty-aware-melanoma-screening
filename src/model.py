"""Step 3: EfficientNet-B0 and focal loss.

- EfficientNet-B0 with ImageNet weights, last layer replaced by a single melanoma logit.
- A switch to keep dropout (and optionally stochastic depth) on at prediction time (MC Dropout).
- Binary focal loss (gamma, alpha from the config).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops import StochasticDepth


def build_model(dropout=0.2, pretrained=True):
    """EfficientNet-B0; the 1000-class ImageNet head is replaced by dropout + 1 output (melanoma logit)."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return model


def enable_mc_dropout(model, stochastic_depth=False):
    """Eval mode (batch norm frozen) but dropout layers stay on, so every pass gives a different answer.

    stochastic_depth=True also keeps StochasticDepth on: the random skipping of residual blocks that
    EfficientNet-B0 uses during training (up to 20% per block). Dropout alone sits only before the last
    layer, so it barely changes the answer; skipping blocks perturbs the whole network.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout) or (stochastic_depth and isinstance(module, StochasticDepth)):
            module.train()
    return model


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017): down-weights easy photos so rare melanomas are not drowned out.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t), where p_t is the probability given to the true class.
    """

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logits, targets = logits.float().view(-1), targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log(p_t)
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()
