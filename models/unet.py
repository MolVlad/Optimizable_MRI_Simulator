import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(self,
                 input_channels, 
                 output_channels,
                 norm_mode,
                 kernel = 3, 
                 stride = 1, 
                 padding = 1,
                 max_pool = True):
        super().__init__()
        self.maxpooling = max_pool
        if norm_mode == 'instance':
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel, stride, padding),
                nn.InstanceNorm2d(num_features = output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel, stride, padding),
                nn.InstanceNorm2d(num_features = output_channels),
                nn.ReLU())
        elif norm_mode == 'batch':
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel, stride, padding),
                nn.BatchNorm2d(num_features = output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel, stride, padding),
                nn.BatchNorm2d(num_features = output_channels),
                nn.ReLU())
        if self.maxpooling:
            self.max_pool = nn.MaxPool2d(kernel_size = 2)
    def forward(self, inputs):
        out = self.block(inputs)
        if self.maxpooling:
            return self.max_pool(out), out
        return out

    
class DecoderBlock(nn.Module):
    def __init__(self, 
                 input_channels,
                 output_channels,
                 norm_mode,
                 kernel = 3,
                 stride = 1,
                 stride_convT = 2,
                 padding = 1,
                 output_pad = 1,
                 input_channels_convT = None,
                 output_channels_convT = None,
                 dropout_proba = 0.5,
                 upsampling_mode = 'conv_transpose'):
        super().__init__()
        if upsampling_mode == 'upsampling':
            self.upsample = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear')
        elif upsampling_mode == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(input_channels_convT, output_channels_convT, kernel, 
                                               stride_convT, padding, output_pad)
        self.dropout = nn.Dropout(dropout_proba)
        
        if norm_mode == 'instance':
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel, stride, padding),
                nn.InstanceNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel, stride, padding),
                nn.InstanceNorm2d(output_channels),
                nn.ReLU())
        elif norm_mode == 'batch':
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU())
        
    def forward(self, inputs, skip):
        out = self.upsample(inputs)
        out = torch.cat([out, skip], dim = 1)
        out = self.dropout(out)
        out = self.block(out)
        return out
        
        
class UNet(nn.Module):
    def __init__(self, 
                 num_classes,
                 min_channels = 32,
                 max_channels = 512, 
                 num_down_blocks = 4, 
                 img_channels = 2, 
                 dropout = 0.5,
                 upsampling_mode = 'conv_transpose',
                 norm_mode = 'instance'):
        super(UNet, self).__init__()
        self.num_down_blocks = num_down_blocks
        self.num_classes = num_classes
        self.output_channels_encoder = [min(min_channels * 2 ** i, max_channels) for i in range(num_down_blocks)]
        self.input_channels_encoder = [img_channels] + self.output_channels_encoder[:-1]
        self.input_channels_decoder = [a + b for a,b in zip(self.output_channels_encoder[::-1], 
                                                            self.input_channels_encoder[::-1])]
        self.output_channels_decoder = self.input_channels_encoder[::-1][:-1]
        self.max_pooling = [True for i in range(num_down_blocks)][:-1] + [False]
        self.convT_input = self.output_channels_encoder[1:][::-1]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_down_blocks):
             self.encoder.append(EncoderBlock(self.input_channels_encoder[i],
                                              self.output_channels_encoder[i],
                                              norm_mode = norm_mode,
                                              max_pool = self.max_pooling[i]))
        for i in range(num_down_blocks - 1):
            if upsampling_mode == 'conv_transpose':
                self.decoder.append(DecoderBlock(self.input_channels_decoder[i],
                                                 self.output_channels_decoder[i],
                                                 norm_mode=norm_mode,
                                                 input_channels_convT = self.convT_input[i],
                                                 output_channels_convT= self.convT_input[i],
                                                 dropout_proba=dropout,
                                                 upsampling_mode = upsampling_mode))
            elif upsampling_mode == 'upsampling':
                self.decoder.append(DecoderBlock(self.input_channels_decoder[i],
                                                 self.output_channels_decoder[i],
                                                 norm_mode=norm_mode,
                                                 dropout_proba=dropout,
                                                 upsampling_mode = upsampling_mode))
        self.last_layer = nn.Sequential(
            nn.Conv2d(min_channels, num_classes, kernel_size = 1),
            nn.Sigmoid())
        # self.last_layer = nn.Conv2d(min_channels, num_classes, kernel_size = 1)
        
    def forward(self, inputs):
        height = inputs.shape[-2]
        width = inputs.shape[-1]
        new_height = round(height / (2 ** self.num_down_blocks)) * 2 ** self.num_down_blocks
        new_width = round(width / (2 ** self.num_down_blocks)) * 2 ** self.num_down_blocks
        new_inputs = F.interpolate(inputs, size = (new_height, new_width))
        
        skip_con = []
        x = new_inputs
        for block in self.encoder:
            res = block(x)
            if isinstance(res, tuple):
                skip_con.append(res[1]) 
                x = res[0]
            else:
                x = res
        for block in self.decoder:
            x = block(x, skip_con.pop())
        logits = self.last_layer(x)
        logits = F.interpolate(logits, size = (height, width))
        return logits