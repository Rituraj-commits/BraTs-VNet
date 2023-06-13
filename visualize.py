from bratsloader import *

from config import *
from losses import *
from metrics import *
from voxelnet import *
from unet3d import *
from vnet import *
from densenet3d import *
from residual_unet3d import *

import torch
import os
from tqdm import tqdm
import SimpleITK as sitk

from torch.autograd import Variable
import matplotlib.pyplot as plt


import warnings

warnings.filterwarnings("ignore")

class ShowResult:
  
    def mask_preprocessing(self, mask):
        """
        Test.
        """
        print(mask[0][0].shape)
        mask_WT = montage(mask[0][0])
        mask_TC = montage(mask[0][1])
        mask_ET = montage(mask[0][2])

        return mask_WT, mask_TC, mask_ET

    def image_preprocessing(self, image):
        """
        Returns image flair as mask for overlaping gt and predictions.
        """
        print(image.shape)
        flair_img = montage(image[0][0])
        return flair_img
    
    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)
        
        fig, axes = plt.subplots(1, 2, figsize = (35, 30))
    
        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
        axes[0].imshow(image, cmap ='bone')
        axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT),
                  cmap='cool_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC),
                  cmap='autumn_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET),
                  cmap='autumn', alpha=0.6)
        axes[1].set_title("Prediction", fontsize=35, weight='bold')
        axes[1].imshow(image, cmap ='bone')
        axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
                  cmap='cool_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
                  cmap='winter', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                  cmap='autumn', alpha=0.6)

        plt.tight_layout()
        
        plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def compute():

    if(args.dataset == "brats"):
        print("Using BraTS Dataset")
        if(args.model == "unet3d"):
            print("Using UNet3D")
            model = Unet3D(c=4, num_classes=3,norm='in')
        elif(args.model == "vnet"):
            print("Using VNet")
            model = VNet(in_channels=4, classes=3)
            model.apply(weights_init)
        elif(args.model == "densevoxelnet"):
            print("Using DenseVoxelNet")
            model = DenseVoxelNet(in_channels=4, classes=3)
        elif(args.model == "densenet"):
            print("Using SkipDenseNet3d")
            model = SkipDenseNet3D(in_channels=4, classes=3)
        elif(args.model == "runet"):
            print("Using ResidualUNet3D")
            model = ResidualUNet3D(in_channels=4, n_classes=3)
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
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
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




output, targets, inputs = compute()


show_result = ShowResult()
show_result.plot(inputs, targets, output)