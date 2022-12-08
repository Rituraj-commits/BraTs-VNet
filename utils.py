import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


  
def mask_preprocessing(mask):
    """
    Test.
    """
    mask = mask.squeeze().cpu().detach().numpy()
    mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

    mask_WT = np.rot90(montage(mask[0]))
    mask_TC = np.rot90(montage(mask[1]))
    mask_ET = np.rot90(montage(mask[2]))

    return mask_WT, mask_TC, mask_ET

def image_preprocessing(image):
    """
    Returns image flair as mask for overlaping gt and predictions.
    """
    image = image.squeeze().cpu().detach().numpy()
    image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
    flair_img = np.rot90(montage(image[0]))
    return flair_img

def plot(image, ground_truth, prediction):
    image = image_preprocessing(image)
    gt_mask_WT, gt_mask_TC, gt_mask_ET = mask_preprocessing(ground_truth)
    pr_mask_WT, pr_mask_TC, pr_mask_ET = mask_preprocessing(prediction)
    
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
                cmap='autumn_r', alpha=0.6)
    axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                cmap='autumn', alpha=0.6)

    plt.tight_layout()
    
    plt.show()