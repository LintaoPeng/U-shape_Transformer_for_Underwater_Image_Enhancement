import torch
from utility import ptcolor as ptcolor
import torch.nn as nn

class lch_Loss(nn.Module):
    def __init__(self, weightC=1,weightH=1,levels=4,eps=0.01,weight=None):
        super(lch_Loss, self).__init__()
        self.weightC=weightC
        self.weightH=weightH
        self.levels=levels
        self.eps=eps
        self.weight=weight


    def hue_to_distribution(self,h, levels, eps=0.0):
        h = h * (levels / 360.0)
        a = torch.arange(levels).float().to(h.device)
        a = a.view(1, levels, 1, 1)
        h=h.unsqueeze(1)
        p = torch.relu(1 - torch.abs(h - a))
        p = p + (a == 0.0).float() * p[:, -1:, :, :]
        p = (p + torch.ones_like(p) * eps) / (1.0 + levels * eps)
        return p



    def forward(self,img,gt):
        img_lch= ptcolor.rgb2lch(img)
        gt_lch= ptcolor.rgb2lch(gt)
        loss_L=torch.mean(torch.abs(img_lch[:,0,:,:]-gt_lch[:,0,:,:]))
        loss_C=torch.mean(torch.abs(img_lch[:,1,:,:]-gt_lch[:,1,:,:]))
        img_H_Dist=torch.clamp(self.hue_to_distribution(img_lch[:,2,:,:],self.levels,self.eps),0.001, 0.999)
        gt_H_Dist =torch.clamp(self.hue_to_distribution(gt_lch[:, 2, :, :], self.levels),0.001, 0.999)
        if self.weight is None:
            loss_H = torch.mean(-torch.mul(gt_H_Dist, torch.log(img_H_Dist)))
        else:
            loss_H = -(gt_lch[:,1,:,:]*(gt_H_Dist*torch.log(img_H_Dist)).sum(1,keepdim=True)).mean()
        loss=loss_L+self.weightC*loss_C+self.weightH*loss_H
        #return(loss,loss_L,loss_C,loss_H)
        return loss
