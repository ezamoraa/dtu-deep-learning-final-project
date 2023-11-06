
# range_image_compression
## data.py

### Tensorflow implementation

Implements the LidarCompressionData class which builds the datasets from training and validation (used in train.py).
To build the dataset, the following steps are performed:

- Get a dataset containing a list of the image file names
- Shuffle the dataset
- Reads uint16 images from the file names and generates tensors of float32 elements
- Performs random cropping of the images to generate the final input tensors. For training, it augments
  the image by concatenating the first "crop_size" colums for padding. Each element is a tuple (image, image),
  indicating that the label is the image itself.
- Groups the input tensors in batches
- Sets the batch shape for each batch (batch_size, crop_size, crop_size, 1)
- Configures prefetching: this allows later elements to be prepared while the current element is being processed.

### Pytorch refactoring strategy

- Create a Pytorch Dataset and use the DataLoader class to generate the batches and suffle the images
- The Dataset class would also perform random cropping
- Set the prefetch_factor in DataLoader

## image_utils.py

### Tensorflow implementation

This file implements the load_image_uint16 functions which is used when building the dataset.

### Pytorch refactoring strategy

- Use torchvision.io.read_image to get the tensors
- Convert the tensor to float using torchvision.transforms.ConvertImageDtype

## utils.py

### Tensorflow implementation

This file implements the load_config_and_model function:

- Creates a model configuration dictionary from a config_map and the model name cmd line arg (string -> cfg_fn -> cfg)
- Overwrites some parameters of the configuration dictionary with the cmd line args
- Optionally enables the XLA (Accelerated Linear Algebra) compiler to optimize models (cfg.xla = True)
- Optionally enables the mixed precision (mixed_float_16) policy
- Configures the multi GPU training strategy
- Creates the model and initializes it using a zero tensor
- Configures the training and validation data loaders taking into account the multi GPU training replicas
- Returns the config and the model objects as a tuple

### Pytorch refactoring strategy

- Keep model configuration dictionary and cmd line arguments
- Might not want to implement XLA JIT compilation optimization since it is more difficult in Pytorch: https://towardsdatascience.com/how-to-accelerate-your-pytorch-training-with-xla-on-aws-3d599bc8f6a9
- Refer to https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html to implement mixed precision (2-3X speedup in some Nvidia architectures)
-

## callbacks.py

### Tensorflow implementation

- CosineLearningRateScheduler callback:
    The learning rate scheduler is a mechanism to update the learning rate on each epoch.
    The cosine learning rate schedule proposed by Ilya et al. (https://arxiv.org/pdf/1608.03983.pdf).
    The learning rate starts to decay starting from epoch self.max_lr_epoch, and reaches the minimum
    learning rate at epoch self.min_lr_epoch.
- Tensorboard callback:
    TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
    The implemented TensorBoard:
        - When ending each training batch: stores the iterations and learning rate
        - When ending each epoch: Applies the model to the first batch in the dataset and stores the input and predicted images

### Pytorch refactoring strategy

- CosineLearningRateScheduler callback:
  - No longer a callback
  - Use the CosineAnnealingWarmRestarts scheduler according to https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2
- Tensorboard callback:
  - Try to add the epoch_end and batch_end metrics to training loop
  - Use torch.utils.tensorboard API (https://pytorch.org/docs/stable/tensorboard.html)

## train.py

### Tensorflow implementation

This file implements the training main program and train function:

- Loads the model using load_config_and_model
- Builds the training and validation datasets, using the same crop size but potentially different batch sizes.
- Creates the cosine learning rate scheduler
- Creates an Adam optimizer
- Creates a model checkpoint callback
- Creates the tensorboard callback
- Optionally loads the weights from a checkpoint
- Compiles the model
- Fits the model
- Saves the model weights using the hdf5 format

### Pytorch refactoring strategy

- Implement standard Pytorch train / validation loop
- Implement weight loading
- Implement model checkpoint according to: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
- Implement final model weights saving (same as checkpoint?)

# range_image_compression/architectures

## Additive LSTM

### Tensorflow implementation

- RnnConv: Convolutional RNN layer
- EncoderRNN: Encoder layer for one iteration (uses Conv2D and RnnConv)
- DecoderRNN: Decoder layer for one iteration (uses Conv2D and RnnConv)
- LidarCompressionNetwork:
    Instantiates the EncoderRNN and DecoderRNN models and performs
    the iterative encoding/decoding algorithm to generate the compression code

### Pytorch refactoring strategy

- Implement standard Pytorch models
- Might need custom implementation for additive LSTM cell
- Also need to add EncoderModel and DecoderModel for inference

# pointcloud_to_range_image/src/architectures/additive_lstm.py

### Tensorflow implementation

- RnnConv: Same as in Python code
- EncoderRNN: Same as in Python code
- DecoderRNN: Same as in Python code
- EncoderModel: Almost the same as LidarCompressionNetwork. Iterative loop changes to generate the codes.
- DecoderModel: Almost the same as LidarCompressionNetwork. Iterative loop changes to generate the images.
- MsgEncoder: ROS node that instantiates the Encoder model. It also encodes the intensity and azimuth images using JPEG.
- MsgDecoder: ROS node that instantiates the Decoder model. It also decodes the intensity and azimuth images using JPEG.

### Pytorch refactoring strategy

- Reuse model classes from range_image_compression Python package
    - With this we can get rid of RnnConv, EncoderRNN, DecoderRNN, EncoderModel and DecoderModel
- Keep MsgEncoder and MsgDecoder classes using the classes from the Python package