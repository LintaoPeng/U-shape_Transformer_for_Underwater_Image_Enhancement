from utility.ptcolor import rgb2lab
from utility.Qnt import quantAB,quantL
import torch
from torch.nn import functional
import torch.nn as nn


class lab_Loss(nn.Module):
    def __init__(self, alpha=1,weight=1,levels=7,vmin=-80,vmax=80):
        super(lab_Loss, self).__init__()
        self.alpha=alpha
        self.weight=weight
        self.levels=levels
        self.vmin=vmin
        self.vmax=vmax

    def Hist_2_Dist_L(self,img, tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2)
        p=functional.softmax(-alpha*img_dist,dim=1)
        return p

    def Hist_2_Dist_AB(self,img,tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2).sum(2)
        p = torch.nn.functional.softmax(-alpha*img_dist, dim=1)
        return p

    def loss_ab(self,img,gt,alpha,tab,levels):
        p= self.Hist_2_Dist_AB(img, tab,alpha).cuda()
        q= self.Hist_2_Dist_AB(gt,tab,alpha).cuda()
        p = torch.clamp(p, 0.001, 0.999)
        loss = -(q*torch.log(p)).sum([1,2,3]).mean()
        return loss




    def forward(self,img,gt):
	    tab=quantAB(self.levels,self.vmin,self.vmax).cuda()
	    lab_img=torch.clamp(rgb2lab(img),self.vmin,self.vmax)
	    lab_gt=torch.clamp(rgb2lab(gt),self.vmin,self.vmax)

	    loss_l=torch.abs(lab_img[:,0,:,:]-lab_gt[:,0,:,:]).mean()
	    loss_AB=self.loss_ab(lab_img[:,1:,:,:],lab_gt[:,1:,:,:],self.alpha,tab,self.levels)
	    loss=loss_l+self.weight*loss_AB
	    #return (loss,loss_l,loss_AB)
	    return loss
