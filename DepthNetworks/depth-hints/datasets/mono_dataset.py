# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image
from numpy.core.fromnumeric import squeeze
from numpy.lib.function_base import flip  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)

import sys
sys.path.append("../..")

from torchattacks.attacks.phy_obj_atk import Phy_obj_atk
from torchattacks.attacks.phy_obj_atk_l0 import Phy_obj_atk_l0
from physicalTrans import PhysicalTrans
from my_utils import ori_H, ori_W, object_dataset_root, train_dist_range

## for lint purpose
# from ....torchattacks.attacks.phy_obj_atk import Phy_obj_atk
# from ....torchattacks.attacks.phy_obj_atk_l0 import Phy_obj_atk_l0
# from ....physicalTrans import PhysicalTrans


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        use_depth_hints
        depth_hint_path
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_depth_hints,
                 depth_hint_path=None,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.use_depth_hints = use_depth_hints

        # assume depth hints npys are stored in data_path/depth_hints unless specified
        if depth_hint_path is None:
            self.depth_hint_path = os.path.join(self.data_path, 'depth_hints')
        else:
            self.depth_hint_path = depth_hint_path

        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.to_pilimage = transforms.ToPILImage()
        self.ori_H = ori_H
        self.ori_W = ori_W
        self.resize_trans = transforms.Resize([self.ori_H, self.ori_W])

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

        # for adv training
        self.is_adv_train = False
        self.load_ben_color = False
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = 1
        side = "l"
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.54
        self.stereo_T = stereo_T

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if 'color_ben' in k:
                 inputs[('color_ben', 0, 0)] = self.to_tensor(color_aug(self.resize[0](frame)))

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


    def set_adv_train(self, model2atk, obj_tensor, mask_tensor, args):
        if args['norm_type'] == "l_inf":
            self.depth_atk = Phy_obj_atk(model2atk, obj_tensor, mask_tensor, 
                                            eps=args['epsilon'], 
                                            alpha=args['alpha'], 
                                            steps=args['step'], dist_range=train_dist_range)
        elif args['norm_type'] == "l_0":
            self.depth_atk = Phy_obj_atk_l0(model2atk, obj_tensor, mask_tensor, 
                                            adam_lr=args["adam_lr"], 
                                            steps=args["step"], 
                                            mask_wt=args["mask_wt"],
                                            l0_thresh=args["l0_thresh"], dist_range=train_dist_range)
        self.load_ben_color = args['load_ben_color']
        self.adv_args = args
        self.is_adv_train = True
        self.obj_mask = mask_tensor.cpu()
        self.obj_img_ben = obj_tensor.cpu()
        self.obj_img_adv = self.obj_img_ben.clone()
        cfg_dict = {'path': f'{object_dataset_root}/training/calib/003086.txt'}
        self.ben_trans = PhysicalTrans(self.obj_img_ben, self.obj_mask, cfg_dict, (1,3,self.ori_H, self.ori_W), dist_range=train_dist_range)
        self.adv_trans = PhysicalTrans(self.obj_img_adv, self.obj_mask, cfg_dict, (1,3,self.ori_H, self.ori_W), dist_range=train_dist_range)
        # intrinscis used in Monodepth2:
        self.adv_K = np.array([[0.58, 0, 0.5, 0],
                      [0, 1.92, 0.5, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        self.adv_K[0, :] *= self.ori_W
        self.adv_K[1, :] *= self.ori_H


    def update_adv_obj(self, scene_imgs):
        """
        This function should be called in each training iteration.
        """
        _, _, _, obj_img_adv = self.depth_atk(scene_imgs, self.adv_args['batch_size'])
        self.obj_img_adv = obj_img_adv.cpu()
        self.adv_trans.reset_img(self.obj_img_adv, self.obj_mask)

    def prep_adv_data(self, inputs, side, do_flip, load_ben_color=False):
        if side == "l":
            l_idx = 0
            r_idx = "s"
        else:
            l_idx = "s"
            r_idx = 0
        color_tensor_l = self.to_tensor(inputs[("color", l_idx, -1)]).unsqueeze(0)
        color_tensor_r = self.to_tensor(inputs[("color", r_idx, -1)]).unsqueeze(0)
        
        if color_tensor_l.size()[2] != self.ori_H or color_tensor_l.size()[3]!= self.ori_W:
            color_tensor_l = self.resize_trans(color_tensor_l)
        if color_tensor_r.size()[2] != self.ori_H or color_tensor_r.size()[3]!= self.ori_W:
            color_tensor_r = self.resize_trans(color_tensor_r)
        # cfg_dict = {'path': f'{object_dataset_root}/training/calib/003086.txt'}
        # ben_trans = PhysicalTrans(self.obj_img_ben, self.obj_mask, cfg_dict, color_tensor_r.size())
        # adv_trans = PhysicalTrans(self.obj_img_adv, self.obj_mask, cfg_dict, color_tensor_l.size())
        
        if side == "l": # keep the current side as adversarial
            # the left image is adversarial
            obj_imgs_out_l, obj_masks_out_l, z0_sample, alpha_sample = self.adv_trans.project(batch_size=1, K=self.adv_K) 
            # the right image is benign
            obj_imgs_out_r, obj_masks_out_r = self.ben_trans.project_w_trans(self.stereo_T, z0_sample=z0_sample, alpha_sample=alpha_sample, K=self.adv_K) 
            # # benign current frame
            # obj_imgs_out_0_ben, obj_masks_out_0_ben, _, _ = self.ben_trans.project(batch_size=1, K=self.adv_K, alpha_sample=alpha_sample, z0_sample=z0_sample) 
        else:
            # the left image is benign
            obj_imgs_out_l, obj_masks_out_l, z0_sample, alpha_sample = self.ben_trans.project(batch_size=1, K=self.adv_K) 
            # the right image is adversarial
            obj_imgs_out_r, obj_masks_out_r = self.adv_trans.project_w_trans(self.stereo_T, z0_sample=z0_sample, alpha_sample=alpha_sample, K=self.adv_K) 
            # # benign current frame
            # obj_imgs_out_0_ben, obj_masks_out_0_ben= self.ben_trans.project_w_trans(self.stereo_T, z0_sample=z0_sample, alpha_sample=alpha_sample, K=self.adv_K) 

        if do_flip:
            obj_imgs_out_l, obj_masks_out_l = torch.flip(obj_imgs_out_l, [3]), torch.flip(obj_masks_out_l, [3])
            obj_imgs_out_r, obj_masks_out_r = torch.flip(obj_imgs_out_r, [3]), torch.flip(obj_masks_out_r, [3])
            # obj_imgs_out_0_ben, obj_masks_out_0_ben = torch.flip(obj_imgs_out_0_ben, [3]), torch.flip(obj_masks_out_0_ben, [3])
        
        color_tensor_l_obj = color_tensor_l * (1 - obj_masks_out_l) + obj_imgs_out_l * obj_masks_out_l
        color_tensor_r_obj = color_tensor_r * (1 - obj_masks_out_r) + obj_imgs_out_r * obj_masks_out_r

        # if side == 'l':
        #     color_tensor_0_obj = color_tensor_l * (1 - obj_masks_out_0_ben) + obj_imgs_out_0_ben * obj_masks_out_0_ben
        # else:
        #     color_tensor_0_obj = color_tensor_r * (1 - obj_masks_out_0_ben) + obj_imgs_out_0_ben * obj_masks_out_0_ben
        
        inputs[("color", l_idx, -1)] = self.to_pilimage(color_tensor_l_obj.squeeze(0))
        inputs[("color", r_idx, -1)] = self.to_pilimage(color_tensor_r_obj.squeeze(0))
        # inputs[("color_ben", 0, -1)] = self.to_pilimage(color_tensor_0_obj.squeeze(0))

        
        if load_ben_color:
            if side == "l":
                # benign current frame
                obj_imgs_out_0_ben, obj_masks_out_0_ben, _, _ = self.ben_trans.project(batch_size=1, K=self.adv_K, alpha_sample=alpha_sample, z0_sample=z0_sample) 
            else:
                # benign current frame
                obj_imgs_out_0_ben, obj_masks_out_0_ben = self.ben_trans.project_w_trans(self.stereo_T, z0_sample=z0_sample, alpha_sample=alpha_sample, K=self.adv_K) 
            if do_flip:
                obj_imgs_out_0_ben, obj_masks_out_0_ben = torch.flip(obj_imgs_out_0_ben, [3]), torch.flip(obj_masks_out_0_ben, [3])
            if side == 'l':
                color_tensor_0_obj = color_tensor_l * (1 - obj_masks_out_0_ben) + obj_imgs_out_0_ben * obj_masks_out_0_ben
            else:
                color_tensor_0_obj = color_tensor_r * (1 - obj_masks_out_0_ben) + obj_imgs_out_0_ben * obj_masks_out_0_ben
            inputs[("color_ben", 0, -1)] = self.to_pilimage(color_tensor_0_obj.squeeze(0))

        # print('do_flip: ', do_flip, "side: ", side)
        # inputs[("color", l_idx, -1)].save('./temp_l_synthesize.jpg')
        # inputs[("color", r_idx, -1)].save('./temp_r_synthesize.jpg')
        # inputs[("color_ben", 0, -1)].save('./temp_0_synthesize_ben.jpg') 
        return inputs

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps
            "depth_hint"                            for depth hint
            "depth_hint_mask"                       for mask of valid depth hints

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # do_color_aug = self.is_train and random.random() > 0.5 and not self.is_adv_train 
        do_color_aug = self.is_train and random.random() > 0.5 and self.adv_args['color_aug']
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
        
        if self.is_adv_train:
            inputs = self.prep_adv_data(inputs, side, do_flip, load_ben_color=self.load_ben_color)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        
        if self.load_ben_color:
            del inputs[("color_ben", 0, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

            # load depth hint
            if self.use_depth_hints:
                side_folder = 'image_02' if side == 'l' else 'image_03'
                depth_folder = os.path.join(self.depth_hint_path, folder, side_folder,
                                            str(frame_index).zfill(10) + '.npy')

                try:
                    depth = np.load(depth_folder)[0]
                except FileNotFoundError:
                    raise FileNotFoundError("Warning - cannot find depth hint for {} {} {}! "
                                            "Either specify the correct path in option "
                                            "--depth_hint_path, or run precompute_depth_hints.py to"
                                            "train with depth hints".format(folder, side_folder,
                                                                            frame_index))

                if do_flip:
                    depth = np.fliplr(depth)

                depth = cv2.resize(depth, dsize=(self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
                inputs['depth_hint'] = torch.from_numpy(depth).float().unsqueeze(0)
                inputs['depth_hint_mask'] = (inputs['depth_hint'] > 0).float()

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
