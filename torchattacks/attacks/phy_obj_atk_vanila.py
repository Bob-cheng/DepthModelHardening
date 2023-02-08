import numpy as np
import torch
import torch.nn as nn
from ..attack import Attack
import PIL.Image as Image
import sys
sys.path.append("../..")
from my_utils import object_dataset_root, ori_W, ori_H
from scipy.ndimage.filters import gaussian_filter
from physicalTrans import PhysicalTrans
from torchvision import transforms
from torchvision.transforms import Resize, ToPILImage, ToTensor
from random import sample
import math
from .light_simulation import tube_light_generation_by_func, simple_add


class Phy_obj_atk_vanila(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, obj_img, obj_mask, dist_range=list(range(5, 31, 2))):
        super().__init__("PGD", model)
        self.obj_img = obj_img
        self.obj_mask = obj_mask
        # self.batch_size = batch_size
        self.depth_target = torch.zeros(1).float().to(self.device)
        self.scene_size = [320, 1024]
        self.resize_trans = Resize(self.scene_size)
        self.eps_for_division = 1e-10

        conf = {}
        conf['path'] = f'{object_dataset_root}/training/calib/003086.txt'  
        self.phy_trans_adv = PhysicalTrans(self.obj_img.clone(), self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)
        self.phy_trans_ben = PhysicalTrans(self.obj_img, self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)


    def forward(self, images, obj_img, batch_size,  cfg_path=f'{object_dataset_root}/training/calib/003086.txt', eval=False):
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

        self.obj_img = obj_img
        self.depth_target = torch.zeros((batch_size, 1, self.scene_size[0], self.scene_size[1])).float().to(self.device)
        obj_img_adv = self.obj_img.clone().detach()
        
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


