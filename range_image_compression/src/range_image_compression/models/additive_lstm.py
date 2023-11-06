from torch import nn

class LidarCompressionNetwork(nn.Module):
    def __init__(self, bottleneck, num_iters, batch_size, input_size):
        self.bottleneck = bottleneck
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.input_size = input_size
        self.beta = 1.0 / self.num_iters

    def forward(self):
        training = self.training
        output = None
        losses = [0]
        loss = sum(losses)*self.beta
        return output, loss