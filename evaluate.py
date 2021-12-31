from vnet import *
from loader import *
from config import *
from metrics import *

import torch
import os

from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def main():
    model = VNet(classes=3, in_channels=4)
    model.apply(weights_init)
    if os.path.exists(args.ModelPath):
        model.load_state_dict(torch.load(args.ModelPath))
        print("Model Loaded")
    else:
        print("Model not found")

    model.cuda()
    model.eval()

    test_dataset = BratsDataset(
        mode="test", crop_dim=(64, 64, 64), dataset_path=args.dataset_path
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)

            dice_score = dice_coef_metric(outputs, labels)
            iou_score = jaccard_coef_metric(outputs, labels)
            dice_scores.append(dice_score)
            iou_scores.append(iou_score)
    print("Dice Score: ", np.nanmean(dice_scores))
    print("IoU Score: ", np.nanmean(iou_scores))


if __name__ == "__main__":
    main()
