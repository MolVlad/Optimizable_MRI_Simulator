import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import os

class MRI(Dataset):
    def __init__(self,
                 phase: str,
                 data_path: str = 'data',
                 mode: str = 'reconstructed_only',
                 num_slices: int = 8,
                 img_size: int = 64,
                 augment = True
                ):
        '''
        reconstructed_only - only one reconstructed img 
        slices - eight slices is concatenated 
        first_plus_reconstr - first slice concatenated to the reconstructed img
        '''
        files = os.listdir(f'{data_path}/{phase}')
        self.phase = phase
        self.masks = []
        self.images = []
        self.img_size = img_size
        self.agument = augment
        threshold = 7 # esimated empirically
        for f in files:
            
            self.masks.append(torch.cat(torch.load(f'{data_path}/{phase}/{f}')['mask'], 
                                        dim = 0).reshape(-1, img_size, img_size))
            
            if mode == 'reconstructed_only':
                zero_slices = torch.cat(torch.load(f'{data_path}/{phase}/{f}')['slices'], 
                                        dim = 0).reshape(-1, 8, img_size, img_size)[:, 0, :, :]
                denoising_masks = zero_slices > threshold
                reconstr = torch.cat(torch.load(f'{data_path}/{phase}/{f}')['reconstruct'], 
                                     dim = 0).reshape(-1, img_size, img_size)
                self.images.append((reconstr * denoising_masks).unsqueeze(1))
                
            elif mode == 'slices':
                self.images.append(torch.cat(torch.load(f'{data_path}/{phase}/{f}')['slices'], 
                                             dim = 0).reshape(-1, 8, img_size, img_size))
                
            elif mode == 'fist_plus_reconstr':
                zero_slices = torch.cat(torch.load(f'{data_path}/{phase}/{f}')['slices'], 
                                        dim = 0).reshape(-1, 8, img_size, img_size)[:, 0, :, :]
                denoising_masks = zero_slices > threshold
                reconstr = torch.cat(torch.load(f'{data_path}/{phase}/{f}')['reconstruct'], 
                                     dim = 0).reshape(-1, img_size, img_size)
                reconstr = reconstr * denoising_masks
                self.images.append(torch.cat([reconstr.unsqueeze(1), zero_slices.unsqueeze(1)], dim = 1))
            elif mode == '1_slice':
                self.images.append(torch.cat(torch.load(f'{data_path}/{phase}/{f}')['slices'], 
                                   dim = 0).reshape(-1, 8, img_size, img_size)[:, 0, :, :].unsqueeze(1))
            elif mode == '3_slices':
                self.images.append(torch.cat(torch.load(f'{data_path}/{phase}/{f}')['slices'],
                                             dim=0).reshape(-1, 8, img_size, img_size)[:,:3,:,:])

        self.masks = torch.cat(self.masks, dim = 0)
        self.images = torch.cat(self.images, dim = 0)
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.Rotate(border_mode=4),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Blur(blur_limit=2),
                A.GridDistortion(p=1, distort_limit=0.1),
                A.Affine(scale=(0.95,1.03), translate_percent=(0,.075))
            ])
        

    def __len__(self):
        return self.masks.shape[0]
    
    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.phase == 'train' and self.agument:
            transformed = self.transform(image = image.permute(1, 2, 0).numpy(), mask = mask.numpy())
            image = transformed['image']
            mask = transformed['mask']
            image = torch.from_numpy(image.copy()).permute(2, 0, 1)
            mask = torch.from_numpy(mask.copy())
        return image, mask.type(torch.LongTensor)