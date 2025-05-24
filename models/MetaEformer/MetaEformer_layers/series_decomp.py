import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if self.kernel_size & 1 == 0:
            front = x[:, 0:1].repeat(1, (self.kernel_size -1 ) // 2)  
            end = x[:, -1:, ].repeat(1, (self.kernel_size) // 2) 
        else:
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2) 
            end = x[:, -1:,].repeat(1, (self.kernel_size - 1) // 2) 
        x = torch.cat([front, x, end], dim=1) 
        # 针对二维的情况
        x = torch.unsqueeze(x, dim=1)
        x = self.avg(x) # [256 * 48]
        x = torch.squeeze(x, dim=1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


