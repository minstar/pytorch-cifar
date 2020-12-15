'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pdb
import copy

from models import *
from utils import progress_bar
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train_batch_size', default=128, type=int, help='comm')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wandb_name', default="", type=str, help='wandb project name')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n_workers', default=1, type=int, help='comm')
parser.add_argument('--alpha', default=0.001, type=float, help='moving rate')
parser.add_argument('--beta', default=0.9, type=float, help='alpha * lr')
parser.add_argument('--tau', default=10, type=int, help='communication period')
parser.add_argument('--rho', default=0.9, type=float, help='momentum')
args = parser.parse_args()

wandb.init(project="EASGD", name=args.wandb_name, config=args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# each local worker should have the same batch size.
assert args.train_batch_size % args.n_workers == 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

master_net = SimpleDLA()
master_net = master_net.to(device)
local_nets = []

for i in range(args.n_workers):
    local_nets += [copy.deepcopy(master_net).to(device)]
    
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

for i in range(args.n_workers):
    if i == 0:
        params =  list(local_nets[i].parameters())
    else:
        params += list(local_nets[i].parameters())

# params += list(master_net.parameters())
optimizer  = optim.SGD(params, lr=args.lr, weight_decay=5e-4, momentum=args.rho) # momentum=0.9
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for i in range(args.n_workers):
        local_nets[i].train()
    master_net.train()
    # for i in range(args.n_workers):
        # local_nets[i].train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bin_size = inputs.size()[0] // args.n_workers

        #communication    
        if (batch_idx + 1) % args.tau == 0:
            sd_master = master_net.state_dict()
            for key in sd_master:
                value = (1 - args.beta) * sd_master[key]
                sd_master[key].copy_(value)
            for i in range(args.n_workers):
                sd_loc = local_nets[i].state_dict()
                sd_cur = copy.deepcopy(sd_loc)
                for key in sd_loc:
                    loc_value    = sd_loc[key] - args.alpha * (sd_loc[key] - sd_master[key])
                    sd_loc[key].copy_(loc_value)
                    master_value = sd_master[key] + args.beta * sd_cur[key] / args.n_workers
                    sd_master[key].copy_(master_value)
                    
        # update each worker
        for num_idx in range(args.n_workers):
            outputs = local_nets[num_idx](inputs[num_idx * bin_size : (num_idx+1) * bin_size, :,:,:])
            loss = criterion(outputs, targets[num_idx * bin_size : (num_idx+1) * bin_size])
            # outputs = master_net(inputs)
            # loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            correct += predicted.eq(targets[num_idx * bin_size : (num_idx+1) * bin_size]).sum().item()      

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    # % (train_loss/(batch_idx+1)/args.n_workers, 100.*correct/total/args.n_workers, correct, total))


def test(epoch):
    global best_acc
    master_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = master_net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({"Test loss": test_loss, "Test acc": 100.*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': master_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        wandb.run.summary["Best Test loss"] = test_loss
        wandb.run.summary["Best Test acc"]  = best_acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
