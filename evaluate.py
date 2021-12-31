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
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def main():
    model = VNet(classes=3, in_channels=4)
    model.apply(weights_init)
    if os.path.exists(args.ModelPath):
        model.load_state_dict(torch.load(args.ModelSavePath))
        print("Model Loaded")
    else:
        print("Model not found")

    model.cuda()
    model.eval()

    test_dataset=BratsDataset(mode='test',crop_dim=(64,64,64),dataset_path=args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    classes = ['WT', 'TC', 'ET']
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)

            dice_scores = dice_coef_metric_per_classes(outputs, labels)
            iou_scores = jaccard_coef_metric_per_classes(outputs, labels)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    with open("TEST_LOGS.txt", "a+") as f:
        f.write("Dice Scores: %s,"%str(dice_scores_per_classes))
        f.write("IoU Scores: %s\n"%str(iou_scores_per_classes))   