#  ==============================================================================
#  MIT License
#  #
#  Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ==============================================================================

class AdditiveLSTMConfig():
    """Configuration for the Additive LSTM Framework"""

    def __init__(self):
        # Data loader
        self.train_data_dir = "demo_samples/training"
        self.val_data_dir = "demo_samples/validation"

        # Training
        self.epochs = 3000
        self.batch_size = 128
        self.val_batch_size = 128
        self.val_freq = 1000
        self.save_freq = 10000
        self.train_output_dir = "output"
        self.xla = True
        self.mixed_precision = False
        self.data_loader_num_workers = 2
        self.data_loader_prefetch_factor = 2

        # Learning Rate scheduler
        self.lr_init = 1e-4
        self.min_learning_rate = 5e-7
        self.min_learning_rate_epoch = self.epochs
        self.max_learning_rate_epoch = 0

        # Network architecture
        self.bottleneck = 32
        self.num_iters = 32
        self.crop_size = 32

        # Give path for checkpoint or set to False otherwise
        self.checkpoint = False
