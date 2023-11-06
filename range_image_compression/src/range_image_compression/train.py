import os
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from .data import LidarCompressionDataset
import models
import configs

g_config_map = {
    "additive_lstm": configs.additive_lstm_cfg.AdditiveLSTMConfig,
}
g_model_map = {
    "additive_lstm": models.additive_lstm.LidarCompressionNetwork,
}

def load_config_and_model(args):
    model_name = args.model.lower()

    cfg = g_config_map[model_name]()

    model = g_model_map[model_name](
        bottleneck=cfg.bottleneck,
        num_iters=cfg.num_iters,
        batch_size=cfg.batch_size,
        input_size=cfg.crop_size
    )
    # init model
    model(torch.zeros((
        cfg.batch_size,
        cfg.crop_size,
        cfg.crop_size,
        1))
    )
    print(model)

    # TODO: Multi-GPU config

    return cfg, model

def train(args):
    cfg, model = load_config_and_model(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

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
                                  batch_size=cfg.train_data_loader_batch_size,
                                  num_workers=cfg.data_loader_num_workers,
                                  prefetch_factor=cfg.data_loader_prefetch_factor,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.val_data_loader_batch_size,
                                num_workers=cfg.data_loader_num_workers,
                                prefetch_factor=cfg.data_loader_prefetch_factor,
                                shuffle=True)

    optimizer = Adam(model.parameters(), lr=cfg.lr_init)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=cfg.max_learning_rate_epoch,
                                            eta_min=cfg.min_learning_rate)
    writer = SummaryWriter()

    step = 0
    start_epoch = 0
    iters = len(train_dataloader)

    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # Que es la vara mae?
    mae = lambda x, y: torch.mean(torch.abs(x - y))

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        for i, input in enumerate(train_dataloader):
            input = input.to(device)
            output, loss = model(input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / iters)
            step += 1

            # Metrics update on each training batch
            writer.add_scalar('Loss/train', loss, step)
            writer.add_scalar('MAE/train', mae(input, output), step)
            writer.add_scalar('Learning Rate/train', scheduler.get_last_lr(), step)

            if step % cfg.val_freq == 0:
                val_maes = []
                val_losses = []
                with torch.no_grad():
                    model.eval()
                    for input in val_dataloader:
                        input = input.to(device)
                        output, loss = model(input)
                        val_losses.append(loss * input.shape[0])
                        val_maes.append(mae(input, output) * input.shape[0])
                    model.train()
                # Metrics update on each validation cycle
                writer.add_scalar('Loss/val', np.sum(val_losses) / len(val_dataset), step)
                writer.add_scalar('MAE/val', np.sum(val_maes) / len(val_dataset), step)

            if step % cfg.save_freq == 0:
                # Save training checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                }, os.path.join(cfg.train_output_dir, f"weights_e={{epoch:05d}}.tar"))

        # Metrics update on each training epoch
        input, _ = next(iter(val_dataloader))
        output = model(input)
        writer.add_image('Images/Input Image', input[0,:,:], epoch)
        writer.add_image('Images/Output Image', output[0,:,:], epoch)

    torch.save(model.state_dict(), os.path.join(args.train_output_dir, 'final_model.pt'))

def parse_args():
    parser = argparse.ArgumentParser(description='Parse Flags for the training script!')
    parser.add_argument('-t', '--train_data_dir', type=str,
                        help='Absolute path to the train dataset')
    parser.add_argument('-v', '--val_data_dir', type=str,
                        help='Absolute path to the validation dataset')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Maximal number of training epochs')
    parser.add_argument('-o', '--train_output_dir', type=str,
                        help="Directory where to write the Tensorboard logs and checkpoints")
    parser.add_argument('-s', '--save_freq', type=int,
                        help="Save freq for model checkpoint")
    parser.add_argument('-m', '--model', type=str, default='additive_lstm',
                        help='Model name either `additive_gru`, `additive_lstm`,'
                             ' `additive_lstm_demo`, `oneshot_lstm`')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()