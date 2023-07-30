import torch
import torch.nn as nn


class Discrimminator(nn.Module):
    def __init__(self):
        super(Discrimminator, self).__init__()

    def forward(self, x):

        return x