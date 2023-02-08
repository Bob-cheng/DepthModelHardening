from math import cos, sin, radians
from operator import matmul
import torch
import numpy as np
from torchvision.transforms import Pad, ToTensor
from torchvision.transforms.functional import perspective
from random import sample
from preprocessing.kitti_util import Calibration
from my_utils import ori_H, ori_W, object_dataset_root

class PhysicalTrans(object):
    def __init__(self, obj_img, obj_mask, cfg, output_size, \
        angle_range=list(range(-30, 31, 5)), dist_range=list(range(5, 10, 2))) -> None:
        """
        obj_img: 1,C,W,H tensor
        obj_mask: 1,1,W,H tensor
        cfg: dictionary, include: fu, fv, cu, cv
        output_size: 4 dimension sequence
        """
        super().__init__()
        self.obj_img = obj_img
        self.obj_mask = obj_mask
        self.cfg = cfg
        self.calib = Calibration(cfg['path'])

        # self.dist_range = list(range(5, 31, 2))
        # self.angle_range = list(range(-30, 31, 5))
        self.dist_range = dist_range
        self.angle_range = angle_range
        self.output_size = output_size 
        # the output size should be: _, _, 375, 1242, otherwise the calibration file cannot be directly used.
        assert output_size[2] == ori_H and output_size[3] == ori_W
        self.padding_img()

        # BMW:      Height: 1.6m,   width: 1.82m 
        # Sedan:    Height: 1.43m,  width: 1.78m 
        # Subaru:   Height: 1.49m,   width: 1.83m 
        # Truck:    Height: 3m,   width: 2.5m 
        # SUV:      Height: 1.77m,   width: 2.00m 
        # TrafficB: Height: 0.75m,  width: 1.5m 
        veh_h = 1.6
        veh_w = 1.82
        cam_h = 1.65
    
        self.x0 = 0
        self.y0 = cam_h - veh_h/2
        self.m = veh_w
        self.n = veh_h

    def fromWorld2Cam(self, location):
        """
        location: [x, y, z] in camera coordinate system, x: right, y: down, z: forward. Unit: meters
        return: [u, v] pixel position in image. u: column index, v: row index. Unit: pixels
        """
        x = location[0]
        y = location[1]
        z = location[2]
        u = int((x - self.cfg['bx']) * self.cfg['fu'] / z + self.cfg['cu'])
        v = int((y - self.cfg['by']) * self.cfg['fv'] / z + self.cfg['cv'])
        return [u, v]

    def objPosOnImage(self, z0, alpha, K=None):
        """
        return: [tl, tr, br, bl] pixel indices
        """
        world_coord = self.fromZA2Coord(z0, alpha)
        # coords_2D = self.calib.project_rect_to_image(world_coord).astype(np.int32)
        if not isinstance(K, type(None)):
            ## Monodepth2 intrinsics
            N = world_coord.shape[0]
            points = np.concatenate((world_coord.T, np.ones((1, N))), axis=0) # 4 * N
            P = K[:3, :] # 3 * 4
            cam_points = np.matmul(P, points) # 3 * N
            pix_coords = cam_points[:2, :] / (cam_points[[2], :] + 1e-7) # 2 * N
            coords_2D = pix_coords.T.astype(np.int32) # N * 2
        else:
            ## Dataset intrinsics
            coords_2D = self.calib.project_rect_to_image(world_coord).astype(np.int32)

        # print('object position on image', coords_2D)
        return coords_2D # N * 2, (u, v)
    
    def fromZA2Coord(self, z0, alpha):
        x_offset = cos(radians(alpha)) * self.m / 2
        x1 = self.x0 - x_offset
        x2 = self.x0 + x_offset
        z_offset = sin(radians(alpha)) * self.m / 2
        zl = z0 - z_offset
        zr = z0 + z_offset
        y1 = self.y0 - self.n / 2
        y2 = self.y0 + self.n / 2
        obj_tl = [x1, y1, zl]
        obj_tr = [x2, y1, zr]
        obj_bl = [x1, y2, zl]
        obj_br = [x2, y2, zr]

        # p1 = self.fromWorld2Cam(obj_tl)
        # p2 = self.fromWorld2Cam(obj_tr)
        # p4 = self.fromWorld2Cam(obj_br)
        # p3 = self.fromWorld2Cam(obj_bl)
        # # print('object position on image', [p1, p2, p4, p3])
        # return [p1, p2, p4, p3]
        
        world_coord = np.array([obj_tl, obj_tr, obj_br, obj_bl])
        return world_coord # 4 * 3

    def padding_img(self):
        _, C, H, W = self.obj_img.size()
        _, _, H_out, W_out = self.output_size
        l_pad = (W_out - W) // 2
        r_pad = W_out - W - l_pad
        t_pad = (H_out - H) // 2
        b_pad = H_out - H - t_pad
        padding_trans = Pad([l_pad, t_pad, r_pad, b_pad])
        self.obj_img_pad = padding_trans(self.obj_img)
        self.obj_mask_pad = padding_trans(self.obj_mask)
        # [u, v]
        tl = [l_pad, t_pad]
        tr = [l_pad + W, t_pad]
        bl = [l_pad, t_pad + H]
        br = [l_pad + W, t_pad + H]
        self.pos_obj_img_start = [tl, tr, br, bl]
        # print('padding position: ', [tl, tr, br, bl])
        
    def reset_img(self, obj_img, obj_mask):
        self.obj_img = obj_img
        self.obj_mask = obj_mask
        self.padding_img()

    def project(self, is_all=False, batch_size=1, z0_sample=None, alpha_sample=None, K=None, rs: np.random.RandomState=None):
        trans_imgs = []
        trans_masks = []
        if is_all:
            z0_sample = []
            alpha_sample = []
            for z0 in self.dist_range:
                for alpha in self.angle_range:
                    z0_sample.append(z0)
                    alpha_sample.append(alpha)
                    pos_obj_img = self.objPosOnImage(z0, alpha, K)
                    trans_img = perspective(self.obj_img_pad, self.pos_obj_img_start, pos_obj_img)
                    trans_mask = perspective(self.obj_mask_pad, self.pos_obj_img_start, pos_obj_img)
                    trans_imgs.append(trans_img)
                    trans_masks.append(trans_mask)
        else:
            if isinstance(z0_sample, type(None)):
                if rs:
                    z0_sample = rs.choice(self.dist_range, batch_size, replace=False)
                else:
                    z0_sample = sample(self.dist_range, batch_size)
            if isinstance(alpha_sample, type(None)):
                if rs:
                    alpha_sample = rs.choice(self.angle_range, batch_size, replace=False)
                else:
                    alpha_sample = sample(self.angle_range, batch_size)
            for i in range(batch_size):
                z0 = z0_sample[i]
                alpha = alpha_sample[i]
                pos_obj_img = self.objPosOnImage(z0, alpha, K)
                trans_img = perspective(self.obj_img_pad, self.pos_obj_img_start, pos_obj_img)
                trans_mask = perspective(self.obj_mask_pad, self.pos_obj_img_start, pos_obj_img)
                trans_imgs.append(trans_img)
                trans_masks.append(trans_mask)
        obj_imgs_out = torch.cat(trans_imgs, dim=0)
        obj_masks_out = torch.cat(trans_masks, dim=0)
        return obj_imgs_out, obj_masks_out, z0_sample, alpha_sample

    def project_w_trans(self, T: np.ndarray, z0_sample, alpha_sample, K=None):
        """
        K: camera intrinsics, 4 * 4
        T: transformation matrics, camera extrinsics, 4 * 4
        """
        trans_imgs = []
        trans_masks = []
        for i in range(len(z0_sample)):
            z0 = z0_sample[i]
            alpha = alpha_sample[i]
            world_coord = self.fromZA2Coord(z0, alpha) # N * 3
            N = world_coord.shape[0]
            points = np.concatenate((world_coord.T, np.ones((1, N))), axis=0) # 4 * N
            if not isinstance(K, type(None)):
                ## Monodepth2 intrinsics
                P = np.matmul(K, T)[:3, :] # 3 * 4
                cam_points = np.matmul(P, points) # 3 * N
                pix_coords = cam_points[:2, :] / (cam_points[[2], :] + 1e-7) # 2 * N
                pos_obj_img = pix_coords.T.astype(np.int32)
            else:
                ## Dataset intrinsics
                pos_obj_img = self.calib.project_rect_to_image(np.matmul(T, points).T[:, :3]).astype(np.int32)
            trans_img = perspective(self.obj_img_pad, self.pos_obj_img_start, pos_obj_img)
            trans_mask = perspective(self.obj_mask_pad, self.pos_obj_img_start, pos_obj_img)
            trans_imgs.append(trans_img)
            trans_masks.append(trans_mask)
        obj_imgs_out = torch.cat(trans_imgs, dim=0)
        obj_masks_out = torch.cat(trans_masks, dim=0)
        return obj_imgs_out, obj_masks_out

if __name__ == "__main__":
    from image_preprocess import process_car_img
    from my_utils import save_pic
    car_img_resize, car_mask_np, _ = process_car_img('BMW.png', '-2')
    print('img size: ', car_img_resize.size, 'mask size: ', car_mask_np.shape)
    img_tensor = ToTensor()(car_img_resize)[:3,:,:].unsqueeze(0).float()
    mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).unsqueeze(0).float()
    print('img tensor size: ', img_tensor.size(), 'mask tensor size: ', mask_tensor.size())

    conf = {}
    # conf['fu'] = 7.215377000000e+02
    # conf['fv'] = 7.215377000000e+02
    # conf['cu'] = 6.095593000000e+02
    # conf['cv'] = 1.728540000000e+02
    # conf['bx'] = 4.485728000000e+01 / (-conf['fu'])  # relative
    # conf['by'] = 2.163791000000e-01  / (-conf['fv'])
    conf['path'] = f'{object_dataset_root}/training/calib/003086.txt'
    phyTrans = PhysicalTrans(img_tensor, mask_tensor, conf, output_size=(1, 3, ori_H, ori_W))

    obj_imgs_out, obj_masks_out, z0_sample, alpha_sample  = phyTrans.project(batch_size=1)

    # z0 = 10
    # alpha = -30
    # pos_obj_img = phyTrans.objPosOnImage(z0, alpha)
    # obj_imgs_out = perspective(phyTrans.obj_img_pad, phyTrans.pos_obj_img_start, pos_obj_img)
    # obj_masks_out = perspective(phyTrans.obj_mask_pad, phyTrans.pos_obj_img_start, pos_obj_img)

    save_pic(obj_imgs_out, 'temp_obj_img_out')
    save_pic(obj_masks_out, 'temp_obj_mask_out')
    save_pic(phyTrans.obj_img_pad, 'temp_pad_obj_img_out')

    stereo_T = np.eye(4, dtype=np.float32)
    baseline_sign = 1
    side = "l"
    side_sign = -1 if side == "l" else 1
    stereo_T[0, 3] = side_sign * baseline_sign * 0.1

    ## intrinscis used in Monodepth2:
    # K = np.array([[0.58, 0, 0.5, 0],
    #               [0, 1.92, 0.5, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]], dtype=np.float32)
    # K[0, :] *= 1242
    # K[1, :] *= 375
    obj_imgs_out_s, obj_masks_out_s = phyTrans.project_w_trans(stereo_T, z0_sample, alpha_sample)

    save_pic(obj_imgs_out_s, 'temp_obj_img_out_s')
    save_pic(obj_masks_out_s, 'temp_obj_mask_out_s')