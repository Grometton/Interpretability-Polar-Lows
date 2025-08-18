import torch
from typing import Dict, List
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator


##########################################################################################
#####                                                                                #####
#####                             FREQUENTIST MODEL                                  #####
#####                                                                                #####
##########################################################################################


# --- Separable Convolution Module ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
# --- Residual Block ---
class XceptionResBlock(nn.Module):
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()

        self.residual = nn.Sequential(SeparableConv2d(input_channels, output_channels), 
                                     nn.BatchNorm2d(output_channels, affine=True),
                                     nn.ReLU(),
                                     SeparableConv2d(output_channels, output_channels), 
                                     nn.BatchNorm2d(output_channels, affine=True),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                     )
        self.shortcut = nn.Conv2d(in_channels=input_channels, 
                                    out_channels=output_channels, 
                                    kernel_size=1, 
                                    stride=2, 
                                    padding=0
                                    )
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    

# --- Customized Xception Model ---
class XceptionCustom(nn.Module): 
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512]):
        super().__init__()
        self.filter_num = filter_num 

        # entry block (conv2d, batch_norm, activation)
        self.entry = nn.Sequential(nn.Conv2d(in_channels=input_channels, 
                                             out_channels=8, 
                                             kernel_size=3, 
                                             stride=2, 
                                             padding=1),
                                   nn.BatchNorm2d(8, affine=True,),
                                   nn.ReLU()
                                  )
        
        # residual block sequence 
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlock(input_channels, out_filters))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        # final separable conv, bn, relu (not present in paper)
        self.final_sepconv = nn.Sequential(SeparableConv2d(self.filter_num[-1], self.filter_num[-1]*2),
                                           nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
                                           nn.ReLU()
                                          )


        # global pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        # classifier 
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), 
                                        nn.Linear(in_features=self.filter_num[-1]*2, out_features=2, bias=False)
                                        )

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten before Linear
        x = self.classifier(x)
        return x
    





##########################################################################################
#####                                                                                #####
#####                               BAYESIAN MODEL                                   #####
#####                                                                                #####
##########################################################################################

        
# --- Separable Convolution Module (BLiTZ: pointwise Bayesian - decide whether to make fully Bayesian later - for now, only pointwise component is Bayesian) ---
class SeparableConv2d_BLITZ(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size=(3,3), stride=1, padding=1, bias=False, prior=None):
        super().__init__()

        # kept depthwise deterministic for now (groups=in_channels - amend later)
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        
        # pointwise: 1x1: make Bayesian with BLiTZ package
        # pass prior dict for customization (to be updated later)
        if prior is None:
            self.pointwise = BayesianConv2d(in_channels, out_channels, kernel_size=(1,1), stride=1, padding=0, bias=bias)
        else:
            self.pointwise = BayesianConv2d(in_channels, out_channels, kernel_size=(1,1), stride=1, padding=0, bias=bias, prior_dist=prior) 

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)   # BLiTZ layer returns a Tensor (samples weights internally)
        return x


# --- Residual Block (uses SeparableConv2d_BLITZ) ---
class XceptionResBlock_BLITZ(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, prior=None):
        super().__init__()

        self.residual = nn.Sequential(SeparableConv2d_BLITZ(input_channels, output_channels, prior=prior),
                                      nn.BatchNorm2d(output_channels, affine=True),
                                      nn.ReLU(),
                                      SeparableConv2d_BLITZ(output_channels, output_channels, prior=prior),
                                      nn.BatchNorm2d(output_channels, affine=True),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
                                     )
        
        # kept shortcut deterministic for now (more stability in theory??) - think whether to swap to BayesianConv2d 
        self.shortcut = nn.Conv2d(in_channels=input_channels, 
                                  out_channels=output_channels, 
                                  kernel_size=(1,1), 
                                  stride=2, 
                                  padding=0)
    
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    


# --- Customized Xception Model ---
@variational_estimator      # adds sample_elbo for training 
class XceptionCustomBLITZ(nn.Module): 
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512], prior=None):
        super().__init__()
        self.filter_num = filter_num 

        # entry block (conv2d, batch_norm, activation) - made Bayesian for now, but to be determined
        self.entry = nn.Sequential(BayesianConv2d(in_channels=input_channels, 
                                             out_channels=8, 
                                             kernel_size=(3,3), 
                                             stride=2, 
                                             padding=1, 
                                             bias=False),  # do we need a bias parameter here??
                                   nn.BatchNorm2d(8, affine=True,),
                                   nn.ReLU()
                                  )
        
        # residual block sequence 
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlock_BLITZ(input_channels, out_filters, prior=prior))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        # final separable conv, bn, relu (not present in paper)
        self.final_sepconv = nn.Sequential(SeparableConv2d_BLITZ(self.filter_num[-1], self.filter_num[-1]*2, prior=prior),
                                           nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
                                           nn.ReLU()
                                          )


        # global pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        # classifier: Dropout + BayesianLinear
        if prior is None:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                BayesianLinear(self.filter_num[-1]*2, 2, bias=False)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                BayesianLinear(self.filter_num[-1]*2, 2, bias=False, prior_dist=prior)
            )

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x