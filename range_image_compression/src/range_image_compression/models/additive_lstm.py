import torch
from torch import nn

class LidarCompressionNetwork(nn.Module):
    def __init__(self, bottleneck, num_iters, batch_size, input_size):
        super().__init__()
        self.bottleneck = bottleneck
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.input_size = input_size
        self.beta = 1.0 / self.num_iters
        self.net = nn.Conv2d(1, 1, 5, padding='same') # TODO: Replace dummy network

    def forward(self, input):
        training = self.training
        # TODO: Replace dummy network
        #------
        output = self.net(input)
        losses = output - 1.0
        #------
        loss = torch.sum(losses)*self.beta
        return output, loss