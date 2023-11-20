# Range Image Compression

This package implements the LIDAR Range Image Compression algorithm using Convolutional RNNs

## Instructions
To generate a virtual environment for the project

```console
$ python3 -m venv .venv
```

To activate the virtual environment:

```console
$ source .venv/bin/activate
```

To install the Python package and its dependencies (the venv should be active):

```console
(.venv) $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(.venv) $ pip3 install -e .
```

To run the training script (run with --help for more details on the arguments):

```console
(.venv) $ range_image_compression_train --train_data_dir ../dataset/pointcloud_compression/train/ --val_data_dir ../dataset/pointcloud_compression/val/ --demo --num_iters 4
```

To start the training script loading a checkpoint file:

```console
(.venv) $ range_image_compression_train --train_data_dir ../dataset/pointcloud_compression/train/ --val_data_dir ../dataset/pointcloud_compression/val/ --demo --num_iters 4 --checkpoint output/weights_step\=20000.tar
```

To run the Tensorboard web server to monitor the training (from same folder where the training script was run):

```console
(.venv) $ tensorboard --logdir runs/
```

Then you can open the web browser at the printed URL (http://localhost:6006/)

To deactivate the virtual environment:

```console
$ deactivate
```