import torch
import torch.nn.functional as F

def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)
    intersection = torch.cat([((preds == i) & (masks == i)).sum((1, 2)).reshape((-1, 1)) for i in range(num_classes)], dim = 1) 
    union = torch.cat([((preds == i) | (masks == i)).sum((1, 2)).reshape((-1, 1)) for i in range(num_classes)], dim = 1) 
    area_sum = torch.cat([(torch.logical_or(preds == i, preds == i).sum(dim=(1,2)).reshape((-1, 1)) + torch.logical_or(masks == i, masks == i).sum(dim=(1,2)).reshape((-1, 1))) for i in range(num_classes)], dim = 1) 
    target = torch.cat([(masks == i).sum((1, 2)).reshape((-1, 1)) for i in range(num_classes)], dim = 1) 
    return intersection, union, target, area_sum

def calc_val_loss(intersection, union, target, area_sum, eps = 1e-7):
    mean_iou = torch.mean((intersection + eps) / (union + eps))
    mean_dice = torch.mean((2 * intersection + eps) / (area_sum + eps))
    mean_class_rec = torch.mean((intersection + eps) / (target + eps))
    mean_acc = torch.mean((intersection.sum(1) + eps) / (target.sum(1) + eps))
    return mean_iou, mean_dice, mean_class_rec, mean_acc


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    Source: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    true = true.unsqueeze(dim=1)

    num_classes = logits.shape[1]
    
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)].to(logits.device)
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)





