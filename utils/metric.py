import torch


# predicted and target should be binarized
def F_score(predicted, target, beta=1.0):
    predicted, target = 1.0 - predicted, 1.0 - target
    true_positive = torch.sum(predicted * target)
    if true_positive == 0:
        return 0, 0, 0
    precision = true_positive / torch.sum(predicted).item()
    recall = true_positive / torch.sum(target).item()
    score = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return score, precision, recall


def Accuracy(predicted, target):
    true = torch.sum(predicted == target).item()
    total = predicted.numel()
    accuracy = true / total
    return accuracy
