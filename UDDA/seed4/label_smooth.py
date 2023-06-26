import warnings

import torch.nn as nn
from torch.nn import functional as F
import torch




class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=3, epsilon=0.15, ):
        super(CE_Label_Smooth_Loss, self).__init__()

        self.classes = classes
        self.epsilon = epsilon


    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss