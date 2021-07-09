import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import dataset
from model import ResNetSimCLR


class Net(nn.Module):
    def __init__(self, num_classes, pretrain_path = ""):
        super(Net, self).__init__()

        # Feature extractor
        self.extractor = ResNetSimCLR(represent_dim = 128)
        if pretrain_path != "":
            self.extractor.load_state_dict(torch.load(pretrain_path))
        self.extractor = self.extractor.f

        # Classifier
        self.classifier = nn.Linear(2048, num_classes, bias = True)


    def forward(self, x):
        x = self.extractor(x)
        feature = torch.flatten(x, start_dim = 1)
        logits = self.classifier(feature)
        return logits


def train_val(net, dataloader, optimizer, device, epoch):
    train_mode = optimizer is not None
    net.train() if train_mode else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(dataloader)
    with (torch.enable_grad() if train_mode else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = net(data)
            loss = loss_criterion(out, target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim = -1, descending = True)
            total_correct_1 += torch.sum((prediction[:, : 1] == target.unsqueeze(dim = -1)).any(dim = -1).float()).item()
            total_correct_5 += torch.sum((prediction[:, : 5] == target.unsqueeze(dim = -1)).any(dim = -1).float()).item()

            data_bar.set_description(f"{'Train' if train_mode else 'Test'} \
                Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f} \
                ACC@1: {total_correct_1 / total_num * 100:.2f}% \
                ACC@5: {total_correct_5 / total_num * 100:.2f}%"
            )
    writer_desc = "Train" if train_mode else "Test"
    writer.add_scalar(writer_desc + "/Loss", total_loss / total_num)
    writer.add_scalar(writer_desc + "/Acc@1", total_correct_1 / total_num * 100)
    writer.add_scalar(writer_desc + "/Acc@5", total_correct_5 / total_num * 100)
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Linear Evaluation')
    parser.add_argument('--model_path', type = str, default = 'results/best.pth',
                        help = 'The pretrained model path')
    parser.add_argument('--batch_size', default = 512, type = int,
                        help = 'Input batch size for training (default: 512)')
    parser.add_argument('--epochs', default = 100, type = int,
                        help = 'Number of epochs to train (default: 100)')
    parser.add_argument("--gpu_device", type = int, default = 0,
                        help = "Select specific GPU to run the model")
    parser.add_argument('--seed', type = int, default = 1, metavar='S',
                        help = 'Random seed (default: 1)')

    writer = SummaryWriter()

    # args parse
    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs

    torch.manual_seed(args.seed)

    # Select the device to train
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_data = CIFAR10(root = 'data', train = True,
        transform = dataset.train_transform, download = True
    )
    train_dataloader = DataLoader(train_data, batch_size = batch_size,
        shuffle = True, num_workers = 4, pin_memory = True, drop_last = True
    )

    test_data = CIFAR10(root = 'data', train = False,
        transform = dataset.test_transform, download = True
    )
    test_loader = DataLoader(test_data, batch_size = batch_size,
        shuffle = False, num_workers = 4, pin_memory = True)

    model = Net(len(train_data.classes), "")
    model.to(device)

    # Fix the feature extractor
    for param in model.extractor.parameters():
        param.requires_grad = False


    optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum = 0.999, weight_decay = 1e-4)
    loss_criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_dataloader, optimizer, device, epoch)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, device, epoch)

        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/downstream_model.pth')