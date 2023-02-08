import numpy as np
import torch
import torch.nn as nn
from ..attack import Attack
import sys
sys.path.append("../..")
from my_utils import object_dataset_root, ori_W, ori_H
from physicalTrans import PhysicalTrans
from torchvision.transforms import Resize
from random import sample


class Phy_obj_atk(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, obj_img, obj_mask, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, dist_range=list(range(5, 31, 2))):
        super().__init__("PGD", model)
        self.obj_img = obj_img
        self.obj_mask = obj_mask
        # self.batch_size = batch_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self._targeted = True
        self.depth_target = torch.zeros(1).float().to(self.device)
        self.scene_size = [320, 1024]
        self.resize_trans = Resize(self.scene_size)

        conf = {}
        conf['path'] = f'{object_dataset_root}/training/calib/003086.txt'  
        self.phy_trans_adv = PhysicalTrans(self.obj_img.clone(), self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)
        self.phy_trans_ben = PhysicalTrans(self.obj_img, self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)


    def forward(self, images, batch_size, cfg_path=f'{object_dataset_root}/training/calib/003086.txt', eval=False):
        r"""
        images: scene image, 1*3*H*W, size: (1242, 375)
        if in eval mode, the fisrt object position and angle sample is fixed instead of random.  
        """

        images = images.detach().to(self.device)
        if images.size()[0] == 1:
            scene_imgs = torch.cat(batch_size * [images.clone()], dim=0)
        elif images.size()[0] == batch_size:
            scene_imgs = images
        else:
            raise RuntimeError('Batch size doesn\'t match!')
        # depth_gt = self.model(images).detach()

        loss = nn.MSELoss()

        obj_img_adv = self.obj_img.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            obj_img_adv = obj_img_adv + torch.empty_like(obj_img_adv).uniform_(-self.eps, self.eps)
            obj_img_adv = torch.clamp(obj_img_adv, min=0, max=1).detach()
        
        self.depth_target = torch.zeros((batch_size, 1, self.scene_size[0], self.scene_size[1])).float().to(self.device)    
        for _ in range(self.steps):
            obj_img_adv.requires_grad_()
            # phy_trans_adv = PhysicalTrans(obj_img_adv, self.obj_mask, conf, images.size())
            self.phy_trans_adv.reset_img(obj_img_adv, self.obj_mask)
            obj_imgs_out_adv, obj_masks_out, _, _ = self.phy_trans_adv.project(batch_size=batch_size)
            adv_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_adv * obj_masks_out
            adv_scenes = self.resize_trans(adv_scenes)
            obj_masks_out = self.resize_trans(obj_masks_out)
            adv_depth = self.model(adv_scenes)
            # print('adv depth size', adv_depth.size())

            cost = -loss(adv_depth * obj_masks_out, self.depth_target)
            # Update adversarial images
            grad = torch.autograd.grad(cost, obj_img_adv,
                                       retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                obj_img_adv = obj_img_adv + self.alpha*grad.sign()
                delta = torch.clamp(obj_img_adv - self.obj_img, min=-self.eps, max=self.eps)
                obj_img_adv = torch.clamp(self.obj_img + delta, min=0, max=1)
                # phy_trans_adv.reset_img(obj_img_adv, self.obj_mask)
        
        # phy_trans_adv = PhysicalTrans(obj_img_adv, self.obj_mask, conf, images.size())
        # phy_trans_ben = PhysicalTrans(self.obj_img, self.obj_mask, conf, images.size())
        self.phy_trans_adv.reset_img(obj_img_adv, self.obj_mask)

        z0_sample = sample(self.phy_trans_ben.dist_range, batch_size)
        alpha_sample = sample(self.phy_trans_ben.angle_range, batch_size)
        if eval:
            z0_sample[0] = 7
            alpha_sample[0] = 0
        
        obj_imgs_out_adv, obj_masks_out, _, _ = self.phy_trans_adv.project(batch_size=batch_size, z0_sample=z0_sample, alpha_sample=alpha_sample)
        adv_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_adv * obj_masks_out
        adv_scenes = self.resize_trans(adv_scenes)

        obj_imgs_out_ben, _, _, _ = self.phy_trans_ben.project(batch_size=batch_size, z0_sample=z0_sample, alpha_sample=alpha_sample)
        ben_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_ben * obj_masks_out
        ben_scenes = self.resize_trans(ben_scenes)
        obj_masks_out = self.resize_trans(obj_masks_out)

        return adv_scenes, ben_scenes, obj_masks_out, obj_img_adv


