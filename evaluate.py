from vnet import *
from unet3d import *
from voxelnet import *
from loader import *
from config import *
from metrics import *

import torch
import os
from tqdm import tqdm

from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()



def compute_scores_per_classes():

    classes = ['WT', 'TC', 'ET']
    if(args.model=="unet3d"):
        print("Using UNet3D")
        model = Unet3D(c=4, num_classes=3)
    elif(args.model=="vnet"):
        print("Using VNet")
        model = VNet(in_channels=4, classes=3)
        model.apply(weights_init)
    elif(args.model=="DenseVoxNet"):
        print("Using DenseVoxelNet")
        model = DenseVoxelNet(in_channels=4,classes=3)
    else:
        raise NotImplementedError

    if os.path.exists(args.ModelPath):
        model.load_state_dict(torch.load(args.ModelPath))
        print("Model Loaded")
    else:
        print("Model not found")

    if(torch.cuda.is_available()):
        print("Using ",torch.cuda.get_device_name(0))
        model.cuda()
    else:
        print("Using CPU")

    model.eval()
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    test_dataset = BratsDataset(
        mode="test", crop_dim=args.crop_dim, dataset_path=args.dataset_path
    )
    dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    with torch.no_grad():
        for _, data in enumerate(tqdm(dataloader)):
            inputs, targets = data
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            logits = model(inputs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])
                iou_scores_per_classes[key].extend(iou_scores[key])

    for key in dice_scores_per_classes.keys():
        print("Mean Dice Score for class {}: {}".format(key, np.nanmean(dice_scores_per_classes[key])))
    
    for key in iou_scores_per_classes.keys():
        print("Mean IoU Score for class {}: {}".format(key, np.nanmean(iou_scores_per_classes[key])))

    


def evaluate():
    if(args.model=="unet3d"):
        print("Using UNet3D")
        model = Unet3D(c=4, num_classes=3)
    elif(args.model=="vnet"):
        print("Using VNet")
        model = VNet(in_channels=4, classes=3)
        model.apply(weights_init)
    elif(args.model=="DenseVoxNet"):
        print("Using DenseVoxelNet")
        model = DenseVoxelNet(in_channels=4,classes=3)
    else:
        raise NotImplementedError

    if os.path.exists(args.ModelPath):
        model.load_state_dict(torch.load(args.ModelPath))
        print("Model Loaded")
    else:
        print("Model not found")

    if(torch.cuda.is_available()):
        print("Using ",torch.cuda.get_device_name(0))
        model.cuda()
    else:
        print("Using CPU")

    model.eval()

    test_dataset = BratsDataset(
        mode="test", crop_dim=args.crop_dim, dataset_path=args.dataset_path
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    dice_scores = []
    iou_scores = []
    st = []
    sp = []
    
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            dice_score = dice_coef_metric(outputs, labels)
            iou_score = jaccard_coef_metric(outputs, labels) 
            sensitivity = Sensitivity(outputs, labels)
            specificity = Specificity(outputs, labels)
            dice_scores.append(dice_score)
            st.append(sensitivity)
            iou_scores.append(iou_score)
            sp.append(specificity)
    print("Mean Dice Score: ", np.nanmean(dice_scores))
    print("Mean IoU Score: ", np.nanmean(iou_scores))
    print("Mean Sensitivity: ", np.nanmean(st))
    print("Mean Specificity: ", np.nanmean(sp))
   

if __name__ == "__main__":
    #compute_scores_per_classes()
    evaluate()
