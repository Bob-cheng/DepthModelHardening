from matplotlib.pyplot import axes, axis
from numpy.core.numeric import zeros_like
from numpy.lib import utils
import torch
import torch.nn as nn
from ..attack import Attack
import sys
sys.path.append("../..")
from my_utils import object_dataset_root, ori_H, ori_W
from physicalTrans import PhysicalTrans
from torchvision.transforms import Resize, ColorJitter
from random import sample
import numpy as np


class Phy_obj_atk_l0(Attack):
    r"""
    Distance Measure : L_0
    """
    def __init__(self, model, obj_img, obj_mask, adam_lr=0.5, steps=10, mask_wt=0.1, l0_thresh=1/10, dist_range=list(range(5, 31, 2))):
        super().__init__("PGD", model)
        self.obj_img = obj_img.clone().detach()
        self.obj_mask = obj_mask.clone().detach()
        # self.batch_size = batch_size
        self.steps = steps
        self.depth_target = torch.zeros(1).float().to(self.device)
        self.scene_size = [320, 1024]
        self.clip_max = 1
        self.learning_rate = adam_lr # adjustable
        self.mask_weight_init = mask_wt # adjustable
        self.mask_weight = self.mask_weight_init
        self.l0_thresh = l0_thresh
        self.l0_clip = self.clip_max / 255.
        self.resize_trans = Resize(self.scene_size)
        self.resize2Ori = Resize([ori_H, ori_W])

        conf = {}
        conf['path'] = f'{object_dataset_root}/training/calib/003086.txt'    
        self.phy_trans_adv = PhysicalTrans(self.obj_img.clone(), self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)
        self.phy_trans_ben = PhysicalTrans(self.obj_img, self.obj_mask, conf, (1,3, ori_H, ori_W), dist_range=dist_range)
        self.color_aug = ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))

    def cal_l0(self):
        threshold = self.l0_clip
        pattern_pos_cur = self.pattern_pos.detach().clone()
        pattern_neg_cur = self.pattern_neg.detach().clone()
        pattern_pos_cur[(pattern_pos_cur < threshold)] = 0
        pattern_neg_cur[(pattern_neg_cur > -threshold)] = 0
        pattern_cur = pattern_pos_cur + pattern_neg_cur
        # l0_cur = np.count_nonzero(np.sum(np.abs(pattern_cur.cpu().numpy()), axis=1))
        l0_cur = torch.count_nonzero(torch.sum(torch.abs(pattern_cur), dim=1))
        return l0_cur

    def forward(self, images, batch_size, cfg_path=f'{object_dataset_root}/training/calib/003086.txt', eval=False, color_jit=False):
        r"""
        images: scene image, 1*3*H*W, size: (1242, 375)
        if in eval mode, the fisrt object position and angle sample is fixed instead of random.  
        """
        img_B,img_C,img_H,img_W = images.size()
        if img_H != ori_H or img_W != ori_W:
            images = self.resize2Ori(images)
            print("image size inconsistent in l0 attack")
        
        images = images.detach().to(self.device)
        if img_B == 1:
            scene_imgs = torch.cat(batch_size * [images.clone()], dim=0)
        elif img_B == batch_size:
            scene_imgs = images
        else:
            raise RuntimeError('Batch size doesn\'t match!')
        # depth_gt = self.model(images).detach()

        for i in range(2):
            init_pattern = np.random.random(self.obj_img.size()) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                self.pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                self.pattern_pos_tensor.requires_grad = True
            else:
                self.pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                self.pattern_neg_tensor.requires_grad = True

        loss = nn.MSELoss()
        optimizer = torch.optim.Adam([self.pattern_pos_tensor, self.pattern_neg_tensor], lr=self.learning_rate, betas=(0.5, 0.9))
        
        self.depth_target = torch.zeros((batch_size, 1, self.scene_size[0], self.scene_size[1])).float().to(self.device)
        l0_norm_init = None

        
        for stp in range(self.steps * 2):
            # update adversatial pattern:
            self.pattern_pos =   torch.clamp(self.pattern_pos_tensor * self.clip_max, min=0.0, max=self.clip_max)
            self.pattern_neg = - torch.clamp(self.pattern_neg_tensor * self.clip_max, min=0.0, max=self.clip_max)
            self.pattern = self.pattern_pos + self.pattern_neg
            
            # compose adv_image
            obj_img_adv = torch.clamp(self.obj_img +  self.pattern, min=0.0, max=self.clip_max)

            # check l0 loss
            l0_norm = self.cal_l0()
            if stp == 0:
                l0_norm_init = l0_norm
            l0_loss_ratio = l0_norm / l0_norm_init
            if l0_loss_ratio <= self.l0_thresh:
                self.mask_weight = 0
                if stp >= self.steps:
                    break
            else:
                self.mask_weight = self.mask_weight_init

            # get adv depth and adv cost
            # phy_trans_adv = PhysicalTrans(obj_img_adv, self.obj_mask, conf, images.size())
            self.phy_trans_adv.reset_img(obj_img_adv, self.obj_mask)
            obj_imgs_out_adv, obj_masks_out, _, _ = self.phy_trans_adv.project(batch_size=batch_size)
            adv_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_adv * obj_masks_out
            adv_scenes = self.resize_trans(adv_scenes)
            adv_obj_mask = self.resize_trans(obj_masks_out)

            # do color augmentation
            if color_jit:
                # if random.random() > 0.5:
                adv_scenes = self.color_aug(adv_scenes)
                
            adv_depth = self.model(adv_scenes)
            adv_cost = loss(adv_depth * adv_obj_mask, self.depth_target)
            
            # get mask cost
            mask_pos = torch.max(torch.tanh(self.pattern_pos_tensor / 10) / (2 - 1e-7) + 0.5, axis=1)[0]
            mask_neg = torch.max(torch.tanh(self.pattern_neg_tensor / 10) / (2 - 1e-7) + 0.5, axis=1)[0]
            mask_cost = torch.mean(mask_pos) + torch.mean(mask_neg)
            
            total_cost = adv_cost + self.mask_weight * mask_cost
            
            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

            # print(f"step: {stp}, adv_cost: {adv_cost}, mask_cost: {mask_cost}, delta_l0_norm: {l0_norm}")
        
        # update adversatial pattern:
        self.pattern_pos =   torch.clamp(self.pattern_pos_tensor * self.clip_max, min=0.0, max=self.clip_max)
        self.pattern_neg = - torch.clamp(self.pattern_neg_tensor * self.clip_max, min=0.0, max=self.clip_max)
        self.pattern_pos[self.pattern_pos < self.l0_clip] = 0
        self.pattern_neg[self.pattern_neg > -self.l0_clip] = 0
        self.pattern = self.pattern_pos + self.pattern_neg
    
        # compose adv_image
        obj_img_adv = torch.clamp(self.obj_img +  self.pattern, min=0.0, max=self.clip_max)

        # phy_trans_adv = PhysicalTrans(obj_img_adv, self.obj_mask, conf, images.size())
        # phy_trans_ben = PhysicalTrans(self.obj_img, self.obj_mask, conf, images.size())
        self.phy_trans_adv.reset_img(obj_img_adv, self.obj_mask)

        z0_sample = sample(self.phy_trans_ben.dist_range, batch_size)
        alpha_sample = sample(self.phy_trans_ben.angle_range, batch_size)
        # print(z0_sample)
        # print(alpha_sample)
        
        if eval:
            z0_sample[0] = 6.1
            alpha_sample[0] = 0
        
        obj_imgs_out_adv, obj_masks_out, _, _ = self.phy_trans_adv.project(batch_size=batch_size, z0_sample=z0_sample, alpha_sample=alpha_sample)
        adv_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_adv * obj_masks_out
        adv_scenes = self.resize_trans(adv_scenes)

        obj_imgs_out_ben, _, _, _ = self.phy_trans_ben.project(batch_size=batch_size, z0_sample=z0_sample, alpha_sample=alpha_sample)
        ben_scenes = scene_imgs * (1 - obj_masks_out) + obj_imgs_out_ben * obj_masks_out
        ben_scenes = self.resize_trans(ben_scenes)
        obj_masks_out = self.resize_trans(obj_masks_out)

        return adv_scenes, ben_scenes, obj_masks_out, obj_img_adv


