import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import dataset
from model import ResNetSimCLR

writer = SummaryWriter()

def train(net, dataloader, optimizer, scheduler, device, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
    for aug_1, aug_2, target in train_bar:
        if device == 'cuda:0':
            aug_1, aug_2 = aug_1.cuda(non_blocking = True), aug_2.cuda(non_blocking = True)

        feature_1, out_1 = net(aug_1)
        feature_2, out_2 = net(aug_2)

        # Shape: [2 * batch, 128]
        out = torch.cat([out_1, out_2], dim = 0)

        # Shape: [2 * batch, 2 * batch]
        dissim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(dissim_matrix) - torch.eye(2 * batch_size, device = dissim_matrix.device)).bool()
        # Shape: [2 * batch, 2 * batch - 1]
        dissim_matrix = dissim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim = -1) / temperature)
        # [2 * batch]
        pos_sim = torch.cat([pos_sim, pos_sim], dim = 0)

        # Compute loss
        loss = (-torch.log(pos_sim / dissim_matrix.sum(dim = -1))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num}')
    scheduler.step()
    writer.add_scalar("train/Loss", train_loss / total_num, epoch)
    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'SimCLR')
    parser.add_argument('--represent_dim', default = 128, type = int,
                        help = 'Feature dim for latent vector')
    parser.add_argument('--temperature', default = 0.5, type = float,
                        help = 'Temperature used in softmax')
    parser.add_argument('--k', default = 200, type = int,
                        help = 'Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default = 512, type = int,
                        help = 'Input batch size for training (default: 512)')
    parser.add_argument('--epochs', default = 500, type = int,
                        help = 'Number of epochs to train (default: 500)')
    parser.add_argument("--gpu_devices", type = int, nargs='+', default = 1,
                        help = "Select specific GPU to run the model")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help = 'Random seed (default: 1)')

    # args parse
    args = parser.parse_args()
    represent_dim, temperature, k = args.represent_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    torch.manual_seed(args.seed)

    # Select the device to train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


    # Prepare data
    train_data = dataset.Cifar10_Pair(root = 'data', train = True,
        transform = dataset.train_transform, download = True
    )
    train_dataloader = DataLoader(train_data, batch_size = batch_size,
        shuffle = True, num_workers = 4, pin_memory = True, drop_last = True
    )

    memory_data = dataset.Cifar10_Pair(root = 'data', train = True,
        transform = dataset.test_transform, download  = True
    )
    memory_loader = DataLoader(memory_data, batch_size = batch_size,
        shuffle = False, num_workers = 4, pin_memory = True)

    test_data = dataset.Cifar10_Pair(root = 'data', train = False,
        transform = dataset.test_transform, download = True
    )
    test_loader = DataLoader(test_data, batch_size = batch_size,
        shuffle = False, num_workers = 4, pin_memory = True)

    # Model and Optimizer setup
    model = ResNetSimCLR(represent_dim)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device, epoch)

