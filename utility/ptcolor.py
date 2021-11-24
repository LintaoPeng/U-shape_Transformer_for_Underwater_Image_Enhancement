
"""Pytorch routines for color conversions and management.

All color arguments are given as 4-dimensional tensors representing
batch of images (Bx3xHxW).  RGB values are supposed to be in the
range 0-1 (but values outside the range are tolerated).

Some examples:

>>> rgb = torch.tensor([0.8, 0.4, 0.2]).view(1, 3, 1, 1)
>>> lab = rgb2lab(rgb)
>>> print(lab.view(-1))
tensor([54.6400, 36.9148, 46.1227])

>>> rgb2 = lab2rgb(lab)
>>> print(rgb2.view(-1))
tensor([0.8000,  0.4000,  0.2000])

>>> rgb3 = torch.tensor([0.1333,0.0549,0.0392]).view(1, 3, 1, 1)
>>> lab3 = rgb2lab(rgb3)
>>> print(lab3.view(-1))
tensor([6.1062,  9.3593,  5.2129])

"""

import torch
from PIL import Image
from torchvision import transforms, utils
import os,sys
import numpy as np


# Helper for the creation of module-global constant tensors
def _t(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)


# Helper for color matrix multiplication
def _mul(coeffs, image):
    coeffs = coeffs.to(image.device).view(3, 3, 1, 1)
    return torch.nn.functional.conv2d(image, coeffs)


_RGB_TO_XYZ = {
    "srgb": _t([[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]),

    "prophoto": _t([[0.7976749, 0.1351917, 0.0313534],
                    [0.2880402, 0.7118741, 0.0000857],
                    [0.0000000, 0.0000000, 0.8252100]])

    }


_XYZ_TO_RGB = {
    "srgb": _t([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]]),

    "prophoto": _t([[ 1.3459433, -0.2556075, -0.0511118],
                    [-0.5445989,  1.5081673,  0.0205351],
                    [0.0000000,  0.0000000,  1.2118128]])
    }


WHITE_POINTS = {item[0]: _t(item[1:]).view(1, 3, 1, 1) for item in [
    ("a", 1.0985, 1.0000, 0.3558),
    ("b", 0.9807, 1.0000, 1.1822),
    ("e", 1.0000, 1.0000, 1.0000),
    ("d50", 0.9642, 1.0000, 0.8251),
    ("d55", 0.9568, 1.0000, 0.9214),
    ("d65", 0.9504, 1.0000, 1.0888),
    ("icc", 0.9642, 1.0000, 0.8249)
]}


_EPSILON = 0.008856
_KAPPA = 903.3
_XYZ_TO_LAB = _t([[0.0, 116.0, 0.], [500.0, -500.0, 0.], [0.0, 200.0, -200.0]])
_LAB_TO_XYZ = _t([[1.0 / 116.0, 1.0 / 500.0, 0], [1.0 / 116.0, 0, 0], [1.0 / 116.0, 0, -1.0 / 200.0]])
_LAB_OFF = _t([16.0, 0.0, 0.0]).view(1, 3, 1, 1)


def apply_gamma(rgb, gamma="srgb"):
    """Linear to gamma rgb.

    Assume that rgb values are in the [0, 1] range (but values outside are tolerated).

    gamma can be "srgb", a real-valued exponent, or None.

    >>> apply_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 0.5).view(-1)
    tensor([0.2500, 0.1600, 0.0100])

    """
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)



def remove_gamma(rgb, gamma="srgb"):
    """Gamma to linear rgb.

    Assume that rgb values are in the [0, 1] range (but values outside are tolerated).

    gamma can be "srgb", a real-valued exponent, or None.

    >>> remove_gamma(apply_gamma(torch.tensor([0.001, 0.3, 0.4])))
    tensor([0.0010,  0.3000,  0.4000])

    >>> remove_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 2.0).view(-1)
    tensor([0.2500, 0.1600, 0.0100])
    """
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma) + \
              torch.min(rgb, rgb.new_tensor(0.0)) # very important to avoid vanishing gradients
        return res


def rgb2xyz(rgb, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """sRGB to XYZ conversion.

    rgb:  Bx3xHxW
    return: Bx3xHxW

    >>> rgb2xyz(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.,  0.,  0.])

    >>> rgb2xyz(torch.tensor([0., 0.75, 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.1868,  0.3737,  0.0623])

    >>> rgb2xyz(torch.tensor([0.4, 0.8, 0.2]).view(1, 3, 1, 1), gamma_correction=None).view(-1)
    tensor([0.4871,  0.6716,  0.2931])

    >>> rgb2xyz(torch.ones(2, 3, 4, 5)).size()
    torch.Size([2, 3, 4, 5])

    >>> xyz2rgb(torch.tensor([-1, 2., 0.]).view(1, 3, 1, 1), clip_rgb=True).view(-1)
    tensor([0.0000,  1.0000,  0.0000])

    >>> rgb2xyz(torch.tensor([0.4, 0.8, 0.2]).view(1, 3, 1, 1), gamma_correction=None, space='prophoto').view(-1)
    tensor([0.4335,  0.6847,  0.1650])

    """
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = remove_gamma(rgb, gamma_correction)
    return _mul(_RGB_TO_XYZ[space], rgb)


def xyz2rgb(xyz, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """XYZ to sRGB conversion.

    rgb:  Bx3xHxW
    return: Bx3xHxW

    >>> xyz2rgb(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.,  0.,  0.])

    >>> xyz2rgb(torch.tensor([0.04, 0.02, 0.05]).view(1, 3, 1, 1)).view(-1)
    tensor([0.3014,  0.0107,  0.2503])

    >>> xyz2rgb(torch.ones(2, 3, 4, 5)).size()
    torch.Size([2, 3, 4, 5])

    >>> xyz2rgb(torch.tensor([-1, 2., 0.]).view(1, 3, 1, 1), clip_rgb=True).view(-1)
    tensor([0.0000,  1.0000,  0.0000])

    """
    rgb = _mul(_XYZ_TO_RGB[space], xyz)
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = apply_gamma(rgb, gamma_correction)
    return rgb


def _lab_f(x):
    x1 = torch.max(x, x.new_tensor(_EPSILON))
    return torch.where(x > _EPSILON, torch.pow(x1, 1.0 / 3), (_KAPPA * x + 16.0) / 116.0)


def xyz2lab(xyz, white_point="d65"):
    """XYZ to Lab conversion.

    xyz: Bx3xHxW
    return: Bx3xHxW

    >>> xyz2lab(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.,  0.,  0.])

    >>> xyz2lab(torch.tensor([0.4, 0.2, 0.1]).view(1, 3, 1, 1)).view(-1)
    tensor([51.8372,  82.3018,  26.7245])

    >>> xyz2lab(torch.tensor([1., 1., 1.]).view(1, 3, 1, 1), white_point="e").view(-1)
    tensor([100., 0., 0.])

    """
    xyz = xyz / WHITE_POINTS[white_point].to(xyz.device)
    f_xyz = _lab_f(xyz)
    return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF.to(xyz.device)


def _inv_lab_f(x):
    x3 = torch.max(x, x.new_tensor(_EPSILON)) ** 3
    return torch.where(x3 > _EPSILON, x3, (116.0 * x - 16.0) / _KAPPA)


def lab2xyz(lab, white_point="d65"):
    """lab to XYZ conversion.

    lab: Bx3xHxW
    return: Bx3xHxW

    >>> lab2xyz(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.,  0.,  0.])

    >>> lab2xyz(torch.tensor([100., 0., 0.]).view(1, 3, 1, 1), white_point="e").view(-1)
    tensor([1.,  1.,  1.])

    >>> lab2xyz(torch.tensor([50., 25., -30.]).view(1, 3, 1, 1)).view(-1)
    tensor([0.2254,  0.1842,  0.4046])

    """
    f_xyz = _mul(_LAB_TO_XYZ, lab + _LAB_OFF.to(lab.device))
    xyz = _inv_lab_f(f_xyz)
    return xyz * WHITE_POINTS[white_point].to(lab.device)


def rgb2lab(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """sRGB to Lab conversion."""
    lab = xyz2lab(rgb2xyz(rgb, gamma_correction, clip_rgb, space), white_point)
    return lab


def lab2rgb(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """Lab to sRGB conversion."""
    return xyz2rgb(lab2xyz(rgb, white_point), gamma_correction, clip_rgb, space)

def lab2lch(lab):
    """Lab to LCH conversion."""
    l = lab[:, 0, :, :]
    c = torch.norm(lab[:, 1:, :, :], 2, 1)
    h = torch.atan2(lab[:, 2, :, :], lab[:, 1, :, :])
    h = h * (180 / 3.141592653589793)
    h = torch.where(h >= 0, h, 360 + h)
    return torch.stack([l, c, h], 1)


def rgb2lch(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """sRGB to LCH conversion."""
    lab = rgb2lab(rgb, white_point, gamma_correction, clip_rgb, space)
    return lab2lch(lab)

def squared_deltaE(lab1, lab2):
    """Squared Delta E (CIE 1976).

    lab1: Bx3xHxW
    lab2: Bx3xHxW
    return: Bx1xHxW

    """
    return torch.sum((lab1 - lab2) ** 2, 1, keepdim=True)


def deltaE(lab1, lab2):
    """Delta E (CIE 1976).

    lab1: Bx3xHxW
    lab2: Bx3xHxW
    return: Bx1xHxW

    >>> lab1 = torch.tensor([100., 75., 50.]).view(1, 3, 1, 1)
    >>> lab2 = torch.tensor([50., 50., 100.]).view(1, 3, 1, 1)
    >>> deltaE(lab1, lab2).item()
    75.0

    """
    return torch.norm(lab1 - lab2, 2, 1, keepdim=True)


def squared_deltaE94(lab1, lab2):
    """Squared Delta E (CIE 1994).

    Default parameters for the 'Graphic Art' version.

    lab1: Bx3xHxW   (reference color)
    lab2: Bx3xHxW   (other color)
    return: Bx1xHxW

    """
    diff_2 = (lab1 - lab2) ** 2
    dl_2 = diff_2[:, 0:1, :, :]
    c1 = torch.norm(lab1[:, 1:3, :, :], 2, 1, keepdim=True)
    c2 = torch.norm(lab2[:, 1:3, :, :], 2, 1, keepdim=True)
    dc_2 = (c1 - c2) ** 2
    dab_2 = torch.sum(diff_2[:, 1:3, :, :], 1, keepdim=True)
    dh_2 = torch.abs(dab_2 - dc_2)
    de_2 = (dl_2 +
            dc_2 / ((1 + 0.045 * c1) ** 2) +
            dh_2 / ((1 + 0.015 * c1) ** 2))
    return de_2


def deltaE94(lab1, lab2):
    """Delta E (CIE 1994).

    Default parameters for the 'Graphic Art' version.

    lab1: Bx3xHxW   (reference color)
    lab2: Bx3xHxW   (other color)
    return: Bx1xHxW

    >>> lab1 = torch.tensor([100., 0., 0.]).view(1, 3, 1, 1)
    >>> lab2 = torch.tensor([80., 0., 0.]).view(1, 3, 1, 1)
    >>> deltaE94(lab1, lab2).item()
    20.0

    >>> lab1 = torch.tensor([100., 0., 0.]).view(1, 3, 1, 1)
    >>> lab2 = torch.tensor([100., 20., 0.]).view(1, 3, 1, 1)
    >>> deltaE94(lab1, lab2).item()
    20.0

    >>> lab1 = torch.tensor([100., 0., 10.]).view(1, 3, 1, 1)
    >>> lab2 = torch.tensor([100., 0., 0.]).view(1, 3, 1, 1)
    >>> round(deltaE94(lab1, lab2).item(), 4)
    6.8966

    >>> lab1 = torch.tensor([100., 75., 50.]).view(1, 3, 1, 1)
    >>> lab2 = torch.tensor([50., 50., 100.]).view(1, 3, 1, 1)
    >>> round(deltaE94(lab1, lab2).item(), 4)
    54.7575

    """
    # The ReLU prevents from NaNs in gradient computation
    sq = torch.nn.functional.relu(squared_deltaE94(lab1, lab2))
    return torch.sqrt(sq)


def _check_conversion(**opts):
    """Verify the conversions on the RGB cube.

    >>> _check_conversion(white_point='d65', gamma_correction='srgb', clip_rgb=False, space='srgb')
    True

    >>> _check_conversion(white_point='d50', gamma_correction=1.8, clip_rgb=False, space='prophoto')
    True

    """
    for r in range(0, 256, 15):
        for g in range(0, 256, 15):
            for b in range(0, 256, 15):
                rgb = torch.tensor([r / 255.0, g / 255.0, b / 255.0]).view(1, 3, 1, 1)
                lab = rgb2lab(rgb, **opts)
                rgb2 = lab2rgb(lab, **opts)
                de = deltaE(rgb, rgb2).item()
                if de > 2e-4:
                    print("Conversion failed for RGB:", r, g, b, " deltaE", de)
                    return False
    return True


def _check_gradients():
    """Verify some borderline gradient computation

    >>> a = torch.zeros(1, 3, 1, 1, requires_grad=True)
    >>> b = torch.zeros(1, 3, 1, 1, requires_grad=True)
    >>> deltaE(a, b).backward()
    >>> torch.any(torch.isnan(a.grad)).item()
    0
    >>> torch.any(torch.isnan(b.grad)).item()
    0

    >>> deltaE94(a, b).backward()
    >>> torch.any(torch.isnan(a.grad)).item()
    0
    >>> torch.any(torch.isnan(b.grad)).item()
    0
    """
    return True


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    print("Test completed")
