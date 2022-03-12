import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils

import numpy as np


class GenderClassifier(nn.Module):
    def __init__(self, args, num_verb):
        super(GenderClassifier, self).__init__()
        print('Build a GenderClassifier Model')

        hid_size = args.hidden_size

        mlp = []
        mlp.append(nn.BatchNorm1d(num_verb))
        mlp.append(nn.Linear(num_verb, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input_rep):

        return self.mlp(input_rep)