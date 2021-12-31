import torch
import numpy as np


def dice_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    scores = []
    probabilities = probabilities.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    num = probabilities.shape[0]
    predictions = probabilities >= threshold
    assert predictions.shape == truth.shape
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:

    scores = []
    probabilities = probabilities.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    num = probabilities.shape[0]
    predictions = probabilities >= treshold
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)
