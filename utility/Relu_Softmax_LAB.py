import torch
from utility import ptcolor as ptcolor
"""
Relu_Softmax_LAB: This collections of methods compute the softmax (on channel L or AB) using the principles:
high value for taller bins and very low values for shorter bins ( almost 0 everywhere).
"""

def softquant(x, vmin, vmax, bins):
    slope = (bins - 1) / (vmax - vmin)
    a = torch.linspace(vmin, vmax, bins, device=x.device)
    diff = (x.unsqueeze(-1) - a).abs()
    return torch.nn.functional.relu(1 - diff * slope) #high value for taller bins, everywhere else almost 0

def softhist_L(x, vmin, vmax, bins):
    x = torch.clamp(x, vmin, vmax)
    q = softquant(x, vmin, vmax, bins)
    return q.view(x.size(0), -1, bins).mean(1)


def softhist_AB(lab, vmax, bins):
    a = torch.clamp(lab[:, 1, :, :], -vmax, vmax)
    b = torch.clamp(lab[:, 2, :, :], -vmax, vmax)
    qa = softquant(a, -vmax, vmax, bins).to(device=lab.device)
    qb = softquant(b, -vmax, vmax, bins).to(lab.device)
    return torch.einsum("bijc,bijd->bcd", qa, qb).to(lab.device) / (a.size(1) * a.size(2))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    im = torch.randn([1,3,64,64])
    lab = ptcolor.rgb2lab(im)
    for c in (0, 1, 2):
        print(lab[:, c, :, :].min(), lab[:, c, :, :].max())
    hist_l = softhist_L(lab[:, 0, :, :], 0, 100, 50)
    print(hist_l.shape)
    hist_ab = softhist_AB(lab, 80, 20)
    plt.plot(hist_l[0].numpy())
    plt.figure()
    plt.imshow(hist_ab[0].numpy())
    plt.show()

