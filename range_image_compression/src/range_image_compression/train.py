import os
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from .data import LidarCompressionDataset
from .configs import additive_lstm_cfg
from .models import additive_lstm

g_config_map = {
    "additive_lstm": additive_lstm_cfg.AdditiveLSTMConfig,
}
g_model_map = {
    "additive_lstm": additive_lstm.LidarCompressionNetwork,
}

def load_config_and_model(args, device):
    print(f"Using device: {device}")
    model_name = args.model.lower()
    cfg = g_config_map[model_name]()

    # overwrite default values in config with parsed arguments
    for key, value in vars(args).items():
        if value:
            setattr(cfg, key, value)

    model = g_model_map[model_name](
        bottleneck=cfg.bottleneck,
        num_iters=cfg.num_iters,
        image_size=(cfg.crop_size,cfg.crop_size),
        device=device,
        demo=cfg.demo,
    )
    model.to(device)
    print(model)

    # Initialize model hidden state
    input = torch.zeros((
        cfg.batch_size,
        1,
        cfg.crop_size,
        cfg.crop_size), device=device)
    model(input)

    # TODO: Multi-GPU config

    return cfg, model

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, model = load_config_and_model(args, device)

    train_dataset = LidarCompressionDataset(
        input_dir=cfg.train_data_dir,
        crop_size=cfg.crop_size,
        augment=True
    )
    val_dataset = LidarCompressionDataset(
        input_dir=cfg.val_data_dir,
        crop_size=cfg.crop_size,
        augment=False
    )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.data_loader_num_workers,
                                  prefetch_factor=cfg.data_loader_prefetch_factor,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.val_batch_size,
                                num_workers=cfg.data_loader_num_workers,
                                prefetch_factor=cfg.data_loader_prefetch_factor,
                                shuffle=True)

    optimizer = Adam(model.parameters(), lr=cfg.lr_init)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=cfg.min_learning_rate_epoch,
                                            eta_min=cfg.min_learning_rate)
    print("Learning rate initialized to {}".format(cfg.lr_init))
    print("Torch log will be save in {}".format(os.path.join(cfg.train_output_dir,"")))
    writer = SummaryWriter(log_dir=os.path.join(cfg.train_output_dir,""))

    step = 0
    start_epoch = 0
    iters = len(train_dataloader)

    if cfg.checkpoint:
        print("Loading checkpoint from {}".format(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']

    # Que es la vara mae?
    mae = lambda x, y: torch.mean(torch.abs(x - y))

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        for i, input in enumerate(train_dataloader):
            input = input.to(device)
            optimizer.zero_grad()
            out = model(input, training=True)
            output, loss = out["outputs"], out["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / iters)
            step += 1

            # Detach tensors from GPU
            input = input.cpu().detach()
            output = output.cpu().detach()
            loss = loss.cpu().detach()

            print(f"Epoch: {epoch}, Batch: {i} (len: {len(input)}), Step: {step}, Loss: {loss}, LR: {scheduler.get_last_lr()[0]:.6f}")

            # Metrics update on each training batch
            writer.add_scalar('Loss/train', loss, step)
            writer.add_scalar('MAE/train', mae(input, output), step)
            writer.add_scalar('Learning Rate/train', scheduler.get_last_lr()[0], step)

            if step % cfg.val_freq == 0:
                print("Validation step")
                val_maes = []
                val_losses = []
                with torch.no_grad():
                    model.eval()
                    for input in val_dataloader:
                        input = input.to(device)
                        out = model(input)
                        output, loss = out["outputs"], out["loss"]

                        # Detach tensors from GPU
                        input = input.cpu().detach()
                        output = output.cpu().detach()
                        loss = loss.cpu().detach()

                        # multiply by batch size to account for different batch sizes
                        val_losses.append(float(loss * input.shape[0]))
                        val_maes.append(float(mae(input, output) * input.shape[0]))
                    model.train()
                # Metrics update on each validation cycle
                writer.add_scalar('Loss/val', np.sum(val_losses) / len(val_dataset), step)
                writer.add_scalar('MAE/val', np.sum(val_maes) / len(val_dataset), step)

            if step % cfg.save_freq == 0:
                print("Saving checkpoint")
                # Make sure that the output directory exists
                os.makedirs(cfg.train_output_dir, exist_ok=True)
                # Save training checkpoint
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                }, os.path.join(cfg.train_output_dir, f"weights_step={step:09d}.tar"))

        # Metrics update on each training epoch
        print("End of epoch: {}".format(epoch))
        input = next(iter(val_dataloader))
        input = input.to(device)
        out = model(input)
        output, loss = out["outputs"], out["loss"]

        # Detach tensors from GPU
        input = input.cpu().detach()
        output = output.cpu().detach()
        loss = loss.cpu().detach()

        writer.add_image('Images/Input Image', input[0,:,:], epoch, dataformats='CHW')
        writer.add_image('Images/Output Image', output[0,:,:], epoch, dataformats='CHW')

    torch.save(model.state_dict(), os.path.join(args.train_output_dir, 'final_model.pt'))

def parse_args():
    parser = argparse.ArgumentParser(description='Parse Flags for the training script!')
    parser.add_argument('-t', '--train_data_dir', type=str,
                        help='Absolute path to the train dataset')
    parser.add_argument('-v', '--val_data_dir', type=str,
                        help='Absolute path to the validation dataset')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Maximal number of training epochs')
    parser.add_argument('-o', '--train_output_dir', type=str, default='output',
                        help="Directory where to write the Tensorboard logs and checkpoints")
    parser.add_argument('-s', '--save_freq', type=int,
                        help="Save freq for model checkpoint")
    parser.add_argument('-m', '--model', type=str, default='additive_lstm',
                        help='Model name either `additive_gru`, `additive_lstm`,'
                             ' `additive_lstm_demo`, `oneshot_lstm`')
    parser.add_argument('-d', '--demo', action='store_true',
                        help="Run in demo mode (smaller model)")
    parser.add_argument('-n', '--num_iters', type=int,
                        help="Number of iterations in the forward pass of the model")
    parser.add_argument('-c', '--checkpoint', type=str,
                        help="Path to checkpoint .tar file to load model from")
    parser.add_argument('-b', '--bottleneck', type=int,
                        help="Size of the bottleneck layer")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()