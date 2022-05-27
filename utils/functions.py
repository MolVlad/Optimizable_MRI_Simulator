import torch
from tqdm import trange
import wandb
from utils.loss import calc_val_data, calc_val_loss, dice_loss



def train_network(network, opt, criterion, num_epochs, train_loader, val_loader, device, saver, use_wandb):

    t = trange(num_epochs, desc='')
    for epoch in t:
        network.train()
        for slices, masks in train_loader:
            opt.zero_grad()
            slices = slices.to(device)
            masks = masks.to(device)
            prediction = network(slices)
            train_loss = criterion(prediction, masks)
            train_loss.backward()
            opt.step()

        network.eval()
        with torch.no_grad():
            for slices, masks in val_loader:
                slices = slices.to(device)
                masks = masks.to(device)
                prediction = network(slices)
                val_loss = criterion(prediction, masks)
                mean_iou, mean_dice, mean_class_rec, mean_acc = calc_val_loss(*calc_val_data(prediction, masks, network.num_classes))
        
        t.set_description(f'Dice:{mean_dice:.3f}', refresh=True)
    
        if use_wandb:
            if epoch % 10 == 0:
                log_images = []
                for s, p, m in zip(slices, prediction, masks):
                    log_images.append(wandb.Image(s[0,:,:].cpu(), 
                                                  masks={
                                                      "predictions" : {
                                                          "mask_data" : p.argmax(axis=0).cpu().numpy()
                                                      },
                                                      "ground_truth" : {
                                                          "mask_data" : m.cpu().numpy(),
                                                      }}))

                wandb.log({'Train loss': train_loss,
                           'Val loss': val_loss,
                           'Mean IOU': mean_iou,
                           'Mean Dice': mean_dice,
                           'Mean class recall': mean_class_rec,
                           'Mean accuracy': mean_acc,
                           'Images': log_images
                          })
            else:
                wandb.log({'Train loss': train_loss,
                           'Val loss': val_loss,
                           'Mean IOU': mean_iou,
                           'Mean Dice': mean_dice,
                           'Mean class recall': mean_class_rec,
                           'Mean accuracy': mean_acc
                          })
            
            # wandb.watch(network)

        if saver:
            saver(mean_dice, epoch, network, opt)

class SaveBestModel:
    def __init__(
        self, path,
        best_metric=0,
    ):
        self.best_metric = best_metric
        self.path = path
        
    def __call__(
        self, current_metric,
        epoch, model, optimizer
    ):
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            # print(f"\nBest metric: {self.best_metric}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, self.path)
            
def mse(img, gt):
    return torch.norm((img.double() - gt.double()), p = 2, dim = [-2, -1]).view(-1, 1)

def psnr(img, gt):
    return 10 * torch.log10(255 / mse(img, gt)) # check MAX I (we have custom float stuff)

def get_threshold(img, gt, values):
    buff = torch.zeros((img.shape[0], len(values)))
    for idx, element in enumerate(values):
        denoising_masks = gt > element
        buff[:, idx] = psnr(img * denoising_masks, gt).squeeze()
    buff = torch.argmax(buff, dim = -1)
    for i in range(buff.shape[-1]):
        buff[i] =  values[buff[i]]
    return buff

def get_denoising_mask(inputs, thresholds):
    denoising_masks = torch.zeros((thresholds.shape[-1], *inputs.shape[-2:]))
    for i in range(thresholds.shape[-1]):
        denoising_masks[i] = torch.gt(inputs[i], thresholds[i])
    return denoising_masks