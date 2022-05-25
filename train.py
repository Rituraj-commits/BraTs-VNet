from vnet import *
from loader import *
from config import *
from losses import *
from metrics import *
from unet3d import *

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import warnings

warnings.filterwarnings("ignore")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def main():
    model = UNet3D(in_channels=4, out_channels=3)
    model.cuda()

    train_dataset = BratsDataset(
        mode="train", crop_dim=args.crop_dim, dataset_path=args.dataset_path
    )

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(args.validation_split * len(train_dataset)))
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
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    criterion = DiceLoss()
    criterion.cuda()
    best_model = 1.0
    tb = SummaryWriter()

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

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0

            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_loss += val_loss.item()
                val_dice += dice_coef_metric(outputs, labels)
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            with open("VAL_LOGS.txt", "a+") as f:
                f.write("epoch: %s," % str(epoch + 1))
                f.write("val_loss: %s," % str(val_loss.detach().cpu().numpy()))
                f.write("val_dice: %s\n" % str(val_dice))

            print("Validation Loss: %.4f, Validation Dice: %.4f" % (val_loss, val_dice))

            tb.add_scalar("Validation/Loss", val_loss, epoch)
            tb.add_scalar("Validation/Dice", val_dice, epoch)
            model.train()
            if val_loss < best_model:
                best_model = val_loss
                torch.save(model.state_dict(), args.ModelSavePath + "best_model.pkl")
                print("Saving best model")

if __name__ == "__main__":
    main()
