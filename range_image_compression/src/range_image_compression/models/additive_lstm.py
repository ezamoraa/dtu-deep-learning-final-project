import torch
from torch import nn, Tensor
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F

from functools import reduce
from operator import __add__


# TODO: Implement the LidarCompressionNetwork class.
# TODO: Make code parameterizable (specifically input size)


def custom_padding(kernel_sizes):
    """
    TODO: What does this do?
    """
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]]) # Returns (left, right, top, bottom)
    conv_padding_2d = (conv_padding[0], conv_padding[2])    # Assumes that the padding is symmetric (same padding on left/right and top/bottom)
    return conv_padding_2d


class RnnConv(nn.Module):
    """Convolutional LSTM cell
    See detail in formula (4-6) in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf
    Args:
        in_channels: the dimensionality of the input space
        out_channels: the dimensionality of the output space
        strides: strides size
        kernel_size: kernel size of convolutional operation
        hidden_kernel_size: kernel size of convolutional operation for hidden state
    Input:
        inputs: input of the layer
        hidden: hidden state (short term memory) [0] and cell state (long term memory) [1] of the layer
    Output:
        new_hidden_state: updated hidden state of the layer (short term)
        new_cell_state: updated cell state of the layer (long term memory)
    """
    def __init__(self, in_channels, out_channels, strides, kernel_size, hidden_kernel_size):
        super(RnnConv, self).__init__()
        self.out_channels = out_channels
        self.strides = strides

        # Initializing the Conv2d layers for input and hidden state
        self.conv_i = nn.Conv2d(in_channels=in_channels,
                                out_channels=self.out_channels * 4,  # times four (4) as we have four (4) gates and each gate has individual weights
                                kernel_size=kernel_size,
                                stride=self.strides,
                                padding=custom_padding(kernel_size),
                                bias=False)

        self.conv_h = nn.Conv2d(in_channels=out_channels,   # TODO: Verify that this is correct.
                                out_channels=self.out_channels * 4,  # times four (4) as we have four (4) gates
                                kernel_size=hidden_kernel_size,
                                stride=(1, 1),
                                padding=custom_padding(hidden_kernel_size),
                                bias=False)
        self.sigmoid_in = nn.Sigmoid()
        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_out = nn.Sigmoid()
        self.tanh_candidate = nn.Tanh()
        self.tanh_newcell = nn.Tanh()

    def forward(self, inputs, hidden):
        # Computing the convolutional operations: input and hidden state
        conv_inputs = self.conv_i(inputs)
        conv_hidden = self.conv_h(hidden[0])    # Short term memory (hidden state)

        # Concatenating the input and hidden state (short term memory) to prepare for the gates:
        sum_conv = conv_inputs + conv_hidden
        # Splitting the sum_conv tensor into the 4 gates. Each gate get the same number of filters.
        in_gate, forget_gate, out_gate, candidate_gate = torch.chunk(sum_conv, 4, dim=1)    # divides into four (4) chunks to split into the four gates.

        # Applying the activation functions
        in_gate = self.sigmoid_in(in_gate)  # input/update gate
        forget_gate = self.sigmoid_f(forget_gate)  # forget gate
        out_gate = self.sigmoid_out(out_gate)  # output gate
        candidate_gate = self.tanh_candidate(candidate_gate)  # candidate / potential cell, calculated from input. Candidate cell is a candidate for updating the cell state

        # Computing new cell (long term memory) and new hidden state (short term memory)
        new_cell_state = forget_gate * hidden[1] + in_gate * candidate_gate
        new_hidden_state = out_gate * self.tanh_newcell(new_cell_state)
        return new_hidden_state, new_cell_state


class EncoderRNN(nn.Module):
    """
    Encoder layer for one iteration.
    Args:
        bottleneck: bottleneck size of the layer (size of latent space, number of bits in the compressed representation?)
    Input:
        input: output array from last iteration.
               In the first iteration, it is the normalized image patch
        hidden2, hidden3, hidden4: hidden and cell states of corresponding ConvLSTM layers
        training: boolean, whether the call is in inference mode or training mode
    Output:
        encoded: encoded binary array in each iteration
        hidden2, hidden3, hidden4: hidden states and cell states of corresponding ConvLSTM layers
    """
    def __init__(self, bottleneck):
        super(EncoderRNN, self).__init__()
        self.bottleneck = bottleneck
        # Define the convolutional layers
        self.Conv_e1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2,2), padding=custom_padding((3,3)), bias=False)    # TODO: stride=(2,2) is not supported in PyTorch when using padding='same'. Can we make a custom padding function?
        self.RnnConv_e1 = RnnConv(in_channels=64, out_channels=256, strides=(2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_e2 = RnnConv(256, 512, (2, 2), (3, 3), (3, 3))
        self.RnnConv_e3 = RnnConv(512, 512, (2, 2), (3, 3), (3, 3))
        self.Conv_b = nn.Conv2d(512, self.bottleneck, kernel_size=(1, 1), bias=False)
        self.tanh_activation = nn.Tanh()
        self.Sign = lambda x: torch.sign(x)

    def forward(self, input, hidden2, hidden3, hidden4, training=False):
        print("EncoderRNN - forward")
        # hidden2, hidden3, and hidden4 are the output of the previous encoder interation. There are 3, because we have 3 layers with one LSTM cell each.
        # Process through the layers sequentially
        # input size (32,32,3)
        x = self.Conv_e1(input)  # First convolutional layer
        # (16,16,64)
        new_hidden2 = self.RnnConv_e1(x, hidden2)   # RnnConv 1
        x = new_hidden2[0]                          # Takes the hidden state (STM) from the RnnConv layer to pass into the next layer
        # (8,8,256)
        new_hidden3 = self.RnnConv_e2(x, hidden3)  # RnnConv 2
        x = new_hidden3[0]
        # (4,4,512)
        new_hidden4 = self.RnnConv_e3(x, hidden4)  # RnnConv 3
        x = new_hidden4[0]
        # (2,2,512)
        # Binarizer:    # TODO: What is this / what does it do?
        x = self.Conv_b(x)  # Final convolutional layer for bottleneck
        x = self.tanh_activation(x)  # tanh activation
        # (2,2,bottleneck)
        if training:
            # Randomized quantization during training
            probs = (1 + x) / 2
            dist = Bernoulli(probs=probs)
            noise = 2 * dist.sample() - 1 - x
            encoded_bitstream = x + torch.stop_gradient(noise)    # TODO: This doesn't work with PyTorch. Check how to do it. https://stackoverflow.com/questions/51529974/tensorflow-stop-gradient-equivalent-in-pytorch
        else:
            encoded_bitstream = self.Sign(x)  # Applying the sign function
        return encoded_bitstream, new_hidden2, new_hidden3, new_hidden4


class DecoderRNN(nn.Module):
    """
    Decoder layer for one iteration.
    Args:
        name: name of decoder layer
    Input:
        input: decoded array in each iteration
        hidden2, hidden3, hidden4, hidden5: hidden and cell states of corresponding ConvLSTM layers
        training: boolean, whether the call is in inference mode or training mode
    Output:
        decoded: decoded array in each iteration
        hidden2, hidden3, hidden4, hidden5: hidden states and cell states of corresponding ConvLSTM layers
    """
    def __init__(self):
        super(DecoderRNN, self).__init__()

        # Define the layers
        self.Conv_d1 = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.RnnConv_d2 = RnnConv(in_channels=512, out_channels=512, strides=(1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d3 = RnnConv(128, 512, (1, 1), (3, 3), (3, 3))
        self.RnnConv_d4 = RnnConv(128, 256, (1, 1), (3, 3), (3, 3))
        self.RnnConv_d5 = RnnConv(64, 128, (1, 1), (3, 3), (3, 3))
        self.Conv_d6 = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=custom_padding((1,1)), bias=False)
        self.tanh_activation = nn.Tanh()
        # Define the depth-to-space (DTS) layers:
        # Rearranges elements in a tensor of shape (C, H, W) to a tensor of shape (C/r², rH, rW), where r is a given upscale factor.
        self.DTS1 = nn.PixelShuffle(2)
        self.DTS2 = nn.PixelShuffle(2)
        self.DTS3 = nn.PixelShuffle(2)
        self.DTS4 = nn.PixelShuffle(2)
        self.Add = nn.ReLU()  # TODO: Fix and make as add in tensorflow
        self.Out = lambda x: x * 0.5

    def forward(self, input, hidden2, hidden3, hidden4, hidden5, training=False):
        print("DecoderRNN - forward")
        # (2,2,bottleneck)
        x_conv = self.Conv_d1(input)  # First convolutional layer
        # (2,2,512)
        hidden2 = self.RnnConv_d2(x_conv, hidden2)  # RnnConv 1
        x = hidden2[0]
        # (2,2,512)
        x = self.Add(x + x_conv)  # Adding and ReLU activation
        x = self.DTS1(x)  # Depth-to-Space
        # (4,4,128)
        hidden3 = self.RnnConv_d3(x, hidden3)  # RnnConv 2
        x = hidden3[0]
        # (4,4,512)
        x = self.DTS2(x)  # Depth-to-Space
        # (8,8,128)
        hidden4 = self.RnnConv_d4(x, hidden4)  # RnnConv 3
        x = hidden4[0]
        # (8,8,256)
        x = self.DTS3(x)  # Depth-to-Space
        # (16,16,64)
        hidden5 = self.RnnConv_d5(x, hidden5)  # RnnConv 4
        x = hidden5[0]
        # (16,16,128)
        x = self.DTS4(x)  # Depth-to-Space
        # (32,32,32)
        # output in range (-0.5,0.5)
        x = self.Conv_d6(x)  # Final convolutional layer
        x = torch.tanh(x)  # tanh activation
        decoded = self.Out(x)  # Output adjustment
        return decoded, hidden2, hidden3, hidden4, hidden5

class LidarCompressionNetwork(nn.Module):
    """
    The model to compress range image projected from point clouds
    The encoder and decoder layers are iteratively called for num_iters iterations.
    This architecture uses additive reconstruction framework and ConvLSTM layers.
    """
    #TODO: Compare with tensorflow implementation.
    def __init__(self, bottleneck, num_iters, batch_size, input_size):
        super().__init__()
        self.bottleneck = bottleneck
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.input_size = input_size
        self.beta = 1.0 / self.num_iters
        # self.net = nn.Conv2d(1, 1, 5, padding='same')

        self.encoder = EncoderRNN(self.bottleneck)
        self.decoder = DecoderRNN()

        # TODO: Figure out what the normalization values 0.1 and 2.5 means and where they come from.
        self.normalize = lambda x: (x-0.1)*2.5

        # TODO: Not really sure how this works - before this was input to the initial_hidden... should it still be?
        self.DIM1 = self.input_size // 2
        self.DIM2 = self.DIM1 // 2
        self.DIM3 = self.DIM2 // 2
        self.DIM4 = self.DIM3 // 2
    
    def compute_loss(self, res):
        """
        Mean Absolute Error Loss function
        """
        loss = torch.mean(torch.abs(res))
        return loss
    
    def initial_hidden(self, batch_size, out_channels, hidden_size):
        """
        Initialize hidden and cell states, all zeros
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        shape = (batch_size, out_channels, hidden_size[0], hidden_size[1])
        hidden = torch.zeros(shape, device = device)    # Make sure device is correct
        cell = torch.zeros(shape, device = device)      # Make sure device is correct
        return hidden, cell

    def forward(self, inputs, training=False):
        # Initialize the hidden states when a new batch comes in
        batch_size = inputs.shape[0]
        hidden_e2 = self.initial_hidden(batch_size, 256, [8, self.DIM2])
        hidden_e3 = self.initial_hidden(batch_size, 512, [4, self.DIM3])
        hidden_e4 = self.initial_hidden(batch_size, 512, [2, self.DIM4])
        hidden_d2 = self.initial_hidden(batch_size, 512, [2, self.DIM4])
        hidden_d3 = self.initial_hidden(batch_size, 512, [4, self.DIM3])
        hidden_d4 = self.initial_hidden(batch_size, 256, [8, self.DIM2])
        hidden_d5 = self.initial_hidden(batch_size, 128, [16,self.DIM1])
        outputs = torch.zeros_like(inputs)

        inputs = self.normalize(inputs)
        res = inputs

        #losses = []
        loss = torch.zeros(1)
        for i in range(self.num_iters):
            print(f"Iteration, LidarCompressionNetwork: {i}")
            code, hidden_e2, hidden_e3, hidden_e4 = self.encoder(res, hidden_e2, hidden_e3, hidden_e4, training=training)

            decoded, hidden_d2, hidden_d3, hidden_d4, hidden_d5 = self.decoder(code, hidden_d2, hidden_d3, hidden_d4, hidden_d5, training=training)

            outputs = outputs + decoded     # Make sure we can just add like this
            # OLD: outputs = tf.add(outputs, decoded)

            # Update res as predicted output in this iteration subtract the original input

            res = outputs - inputs  # Make sure we can just subtract like this
            # OLD: res = self.subtract([outputs, inputs])...

            #losses.append(self.compute_loss(res))
            loss += self.compute_loss(res)

        loss = loss * self.beta
        #losses = torch.FloatTensor(losses)  # TODO: make this a tensor from the beginning?
        #loss = torch.sum(losses)*self.beta

        # Denormalize the tensors and convert to float32
        outputs = torch.clamp(((outputs * 0.4) + 0.1), min=0, max=1)
        # OLD: outputs = tf.clip_by_value(tf.add(tf.multiply(outputs, 0.4), 0.1), 0, 1)

        outputs = outputs.type(torch.FloatTensor)
        # OLD: outputs = tf.cast(outputs, dtype=tf.float32)
        return outputs, loss