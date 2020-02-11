import torch
from torch.nn import BCELoss
from torch import nn


def focal_loss(predicted, target, alpha=0.70, power=2):
    criterion = BCELoss(reduction="none")
    loss = criterion(predicted, target)
    loss = alpha * loss * (1 - target) * (predicted ** power) + (1 - alpha) * loss * target * ((1 - predicted) ** power)
    return loss.mean()


def dice_loss(predicted, target):
    predicted = 1 - predicted
    target = 1 - target

    dice = 2 * torch.sum(predicted * target) / (torch.sum(predicted * predicted) + torch.sum(target * target))
    return 1 - dice


def fscore_loss(predicted, target, beta=1):
    # boundary is black(0), first convert boundary as positive class(1)
    predicted = 1 - predicted
    target = 1 - target

    tp = torch.sum(predicted * target)
    fp = torch.sum(predicted * (1 - target))
    fn = torch.sum((1 - predicted) * target)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    fscore = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall + epsilon)

    return 1 - fscore


def near_edge_loss(predicted, target, kernel_size=3):
    smoothing_kernel = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1,
                                 padding=(kernel_size-1) // 2, bias=False).cuda()
    smoothing_kernel.weight.data = torch.ones(kernel_size, kernel_size).float().unsqueeze(0).unsqueeze(0).cuda() / \
                                   float(kernel_size * kernel_size)
    smoothing_kernel.weight.requires_grad = False
    smoothed_target = smoothing_kernel(target)
    smoothed_target[smoothed_target < 0.98] = 0
    smoothed_target[smoothed_target != 0] = 1.0
    near_edge_target = torch.zeros_like(smoothed_target).cuda().float()
    near_edge_target[smoothed_target != target] = 1

    predicted = predicted * near_edge_target
    loss = binary_cross_entropy_loss(predicted, near_edge_target)

    return loss


def binary_cross_entropy_loss(predicted, target):
    criterion = BCELoss()
    loss = criterion(predicted, target)
    return loss


def weighted_bce_loss(predicted, target, weight):
    loss = - weight * (1 - target) * torch.log(1 - predicted) - (1 - weight) * target * torch.log(predicted)
    return loss.mean()
