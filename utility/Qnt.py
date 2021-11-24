import torch
import numpy as np
import itertools
"""
Qnt: Quantization Methods. this collection of methods, compute the quantization tables  for RGB, and LAB color space. 
These methods are organized in a way that each bin is recognized by its central value.
ES: RGB 0-255
    2 levels of quantization for each channel
    0-127:128-255. these two intervals are represented as 63.5 and 191.5 in the table.
    then, only for RGB, the values are normalized in 0-1 range
    
    For L and AB the methods provide in output the 2D and 1D quantization tables. The values are not normalized.
"""
def quantRGB(bins,vmax=255,vmin=0):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat=torch.cartesian_prod(a,a,a)/vmax
    return mat.view(1,bins**3,3,1,1)

def quantL(bins,vmax,vmin):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat = a
    return mat.view(1,bins,1,1)


def quantAB(bins, vmax,vmin):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat=torch.cartesian_prod(a,a)
    return mat.view(1,bins**2,2,1,1)







