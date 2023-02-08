import torch
import torch.nn as nn
import numpy as np


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=1000, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder #(num_classes=dim, zero_init_residual=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.my_cos=nn.CosineSimilarity()
        # print(self.encoder)
        # build a 3-layer projector
        prev_dim = 512
        # prev_dim = base_encoder.fc.weight.shape[1]
        

        # self.encoder.encoder.fc = nn.Sequential(
        #                                 self.encoder.encoder.fc,
        #                                 nn.BatchNorm1d(prev_dim, affine=False)) # output layer
        # self.encoder.encoder.fc[0].bias.requires_grad = False # hack: not use bias as it is followed by BN
        # print(self.encoder.encoder.fc[0])



        # self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, feature1, feature2): # adv_color, ben_color
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        x1, x2 = feature1[-1], feature2[-1]
        
        # compute features for one view
        z1 = self.avgpool(x1)
        z1 = z1.reshape(z1.shape[0],z1.shape[1])

        z2 = self.avgpool(x2)
        z2 = z2.reshape(z2.shape[0],z2.shape[1])
        # print(z1.size())

        z1 = self.projector(z1) # NxC
        z2 = self.projector(z2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        
        p1,p2,z1,z2 = p1,p2,z1.detach(),z2.detach()

        contrastive_loss = -(self.my_cos(p1, z2).mean() + self.my_cos(p2, z1).mean()) * 0.5

        return contrastive_loss