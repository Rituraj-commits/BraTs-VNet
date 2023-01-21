from vnet import *
from unet3d import *
from voxelnet import *
from loader import *
from config import *
from metrics import *
from utils import *

import torch
import os
from tqdm import tqdm

from torch.autograd import Variable


import warnings

warnings.filterwarnings("ignore")


def compute():

    if(args.model=="unet3d"):
        print("Using UNet3D")
        model = Unet3D(c=4, num_classes=3)
    elif(args.model=="vnet"):
        print("Using VNet")
        model = VNet(in_channels=4, classes=3)
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
    results = {"image": [], "GT": [],"Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
                inputs, targets = data
                inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
                output = model(inputs)
                output = torch.sigmoid(output)
                output = (output.detach().cpu().numpy()>0.5)
                targets = targets.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                return output, targets, inputs
