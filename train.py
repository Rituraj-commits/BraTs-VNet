from loader import *
from config import *
from losses import *
from metrics import *
from voxelnet import *
from unet3d import *
from vnet import *

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import warnings

warnings.filterwarnings("ignore")

from random import seed

seed(args.seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def main():
    if(args.model=="unet3d"):
        print("Using UNet3D")
        model = Unet3D(c=4, num_classes=3)
    elif(args.model=="vnet"):
        print("Using VNet")
        model = VNet(in_channels=4, classes=3)
        model.apply(weights_init)
    elif(args.model=="densevoxelnet"):
        print("Using DenseVoxelNet")
        model = DenseVoxelNet(in_channels=4,classes=3)
    else:
        raise NotImplementedError

    if(torch.cuda.is_available()):
        print("Using ",torch.cuda.get_device_name(0))
        model.cuda()
    else:
        print("Using CPU")

    train_dataset = BratsDataset(
        mode="train", crop_dim=args.crop_dim, dataset_path=args.dataset_path
    )

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(train_dataset)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, sampler=train_sampler
    )
    val_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, sampler=valid_sampler
    )

    if args.optimizer == "adam":
        print("Using Adam Optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        print("Using SGD Optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        raise NotImplementedError

    if args.loss == "dice":
        print("Using Dice Loss")
        criterion = DiceLoss()
    elif args.loss == "bce":
        print("Using BCE Loss")
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "hausdorff":
        print("Using Hausdorff Loss")
        criterion = HausdorffDTLoss()
    else:
        raise NotImplementedError

    criterion.cuda()
    

    if not os.path.isdir(args.ModelSavePath):
        os.makedirs(args.ModelSavePath)

    print("Start Training")
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                    % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item())
                )

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), args.ModelSavePath + "%s_model_%s.pkl" % (args.model, args.loss))
            print("Saving best model")


if __name__ == "__main__":
    main()
