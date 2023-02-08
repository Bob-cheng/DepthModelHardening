from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import random

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
sys.path.append('../..')

# from ...depth_model import DepthModelWrapper
from depth_model import DepthModelWrapper, import_depth_model
from torchattacks.attacks.phy_obj_atk import Phy_obj_atk
from torchattacks.attacks.phy_obj_atk_l0 import Phy_obj_atk_l0
from torchattacks.attacks.phy_obj_atk_l2 import Phy_obj_atk_l2
from torchattacks.attacks.phy_obj_atk_apgd import Phy_obj_atk_APGD
from torchattacks.attacks.phy_obj_atk_square import Phy_obj_atk_Square
from torchattacks.attacks.phy_obj_atk_arbi import Phy_obj_atk_arbi
from torchattacks.attacks.pgd_depth import PGD_depth
from image_preprocess import process_car_img
from dataLoader import KittiLoader
from my_utils import object_dataset_root, ori_W, ori_H, eval_depth_diff, get_mean_depth_diff, save_pic, eval_depth_diff_jcl

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(17)

def compute_errors(gt, pred, mask=None):
    """Computation of error metrics between predicted and ground truth depths
    """
    if isinstance(mask, type(None)):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_err = np.mean(np.abs(gt - pred))

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)
    else:
        assert mask.shape == gt.shape and mask.shape == pred.shape
        total_pixel = mask.sum()
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = ((thresh < 1.25     ) * mask).sum() / total_pixel
        a2 = ((thresh < 1.25 ** 2) * mask).sum() / total_pixel
        a3 = ((thresh < 1.25 ** 3) * mask).sum() / total_pixel

        abs_err = (np.abs(gt - pred)) * mask
        abs_err = abs_err.sum() / total_pixel
        
        rmse = ((gt - pred) ** 2) * mask
        rmse = np.sqrt(rmse.sum() / total_pixel)

        rmse_log = ((np.log(gt) - np.log(pred)) ** 2) * mask
        rmse_log = np.sqrt(rmse_log.sum() / total_pixel)

        abs_rel = np.sum(np.abs(gt - pred) / gt * mask) / total_pixel

        sq_rel = np.sum(((gt - pred) ** 2) / gt * mask) / total_pixel


    return abs_err, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate_attacks(model2atk: DepthModelWrapper, args, eval_count=25):
    print(args)
    car_img_resize, car_mask_np, _ = process_car_img('BMW.png', '-2')
    obj_tensor = transforms.ToTensor()(car_img_resize)[:3,:,:].unsqueeze(0).float().cuda()
    mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).unsqueeze(0).float().cuda()
            
    if args['norm_type'] == "l_inf":
        depth_atk = Phy_obj_atk(model2atk, obj_tensor, mask_tensor, 
                                        eps=args['epsilon'], 
                                        alpha=args['alpha'], 
                                        steps=args['step'])
    elif args['norm_type'] == "l_0":
        depth_atk = Phy_obj_atk_l0(model2atk, obj_tensor, mask_tensor, 
                                        adam_lr=args["adam_lr"], 
                                        steps=args["step"], 
                                        mask_wt=args["mask_wt"],
                                        l0_thresh=args["l0_thresh"])
    elif args['norm_type'] == "image":
        depth_atk = PGD_depth(model2atk, eps=args['epsilon'], alpha=args['alpha'],steps=args['step'])
        depth_atk._targeted = True
    elif args['norm_type'] == "l_2":
        depth_atk = Phy_obj_atk_l2(model2atk, obj_tensor, mask_tensor, 
                                        eps=args['epsilon'], 
                                        alpha=args['alpha'], 
                                        steps=args['step'])
    elif args['norm_type'] == "APGD":
        depth_atk = Phy_obj_atk_APGD(model2atk, obj_tensor, mask_tensor, 
                                        eps=args['epsilon'], 
                                        steps=args['step'])
    elif args['norm_type'] == "Square":
        depth_atk = Phy_obj_atk_Square(model2atk, obj_tensor, mask_tensor, 
                                        eps=args['epsilon'], 
                                        n_queries=args['n_queries'])
    elif args['norm_type'] == 'arbi':
        depth_atk = Phy_obj_atk_arbi(model2atk, obj_tensor, mask_tensor)


    kitti_loader_test = KittiLoader(mode='val', root_dir=object_dataset_root, train_list='trainval.txt', val_list='trainval.txt', size=(ori_W, ori_H))
    test_loader = DataLoader(kitti_loader_test, batch_size=args['batch_size'], shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    atk_perf = 0
    start_idx = 42
    errors = []
    loader_iter = iter(test_loader)
    i = -1
    while True:
        try:
            (scene_img, _) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_loader)
            (scene_img, _) = next(loader_iter)
        i += 1
        if i < start_idx:
            continue
        if i - start_idx >= eval_count:
            break
        # print("evaluating scene index: ", i)
        scene_img = scene_img.cuda()
        # scene_img = trans(scene_img)
        if args['norm_type'] == "image":
            adv_images, ben_images = depth_atk(scene_img)
            obj_masks_out = None
        else:
            adv_images, ben_images, obj_masks_out, obj_img_adv = depth_atk(scene_img, args['batch_size'], eval=True)
        
        with torch.no_grad():
            disp_gt = model2atk(ben_images)
            disp_atk = model2atk(adv_images)
        if i == start_idx:
            result_img_atk, _, _ = eval_depth_diff(adv_images[[0]], ben_images[[0]], model2atk, disp1=disp_atk[[0]], disp2=disp_gt[[0]])
        # atk_perf += get_mean_depth_diff(disp_atk, disp_gt, scene_car_mask=obj_masks_out, use_abs=True) # the higher, the better attack, the lower robustness
        
        gt_depth = torch.clamp(disp_to_depth(torch.abs(disp_gt),0.1,100)[1]*STEREO_SCALE_FACTOR,max=MAX_DEPTH, min=MIN_DEPTH)
        atk_depth = torch.clamp(disp_to_depth(torch.abs(disp_atk),0.1,100)[1]*STEREO_SCALE_FACTOR,max=MAX_DEPTH, min=MIN_DEPTH)

        obj_masks_out = obj_masks_out.cpu().numpy() if obj_masks_out != None else None
        errors.append(compute_errors(gt_depth.cpu().numpy(), atk_depth.cpu().numpy(), mask=obj_masks_out))
    
    # visualize_perterbation(obj_tensor, obj_img_adv)
    # result_img_atk.save('./temp_atk_perform.png')
    # atk_perf = atk_perf / eval_count
    # print("attack performance: mean depth estimation eror: ", atk_perf)

    mean_errors = np.array(errors).mean(0)
    max_errors = np.array(errors).max(0)

    print("Mean Error:")
    print("\n  " + ("{:>8} | " * 8).format("abs_err", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("Max Error:")
    print("\n  " + ("{:>8} | " * 8).format("abs_err", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*max_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    return mean_errors


def import_model(opt, model_type='monodepth2'):
    if model_type == 'manydepth':
        depth_model = import_depth_model((1024, 320), model_type='manydepth', pre_model_path=opt.load_weights_folder)
        encoder = depth_model.encoder
        depth_decoder = depth_model.decoder
        encoder_dict =depth_model.encoder_dict
        depth_model.cuda()
        depth_model.eval()
    else:
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        depth_model = DepthModelWrapper(encoder, depth_decoder)
        depth_model.cuda()
        depth_model.eval()
    return encoder, depth_decoder, depth_model, encoder_dict

def visualize_perterbation(obj_tensor, adv_obj):
    perturbation = torch.clamp(torch.abs(obj_tensor - adv_obj) * 5, 0, 1)
    save_pic(obj_tensor, 0)
    save_pic(adv_obj, 1)
    save_pic(perturbation, 2)

def evaluate(opt, encoder, depth_decoder, encoder_dict):
    """Evaluates a pretrained model using a specified test set
    """
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        # decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        # encoder_dict = torch.load(encoder_path)


        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        # model_dict = encoder.state_dict()
        # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # depth_decoder.load_state_dict(torch.load(decoder_path))

        # encoder.cuda()
        # encoder.eval()
        # depth_decoder.cuda()
        # depth_decoder.eval()

        # depth_model = DepthModelWrapper(encoder, depth_decoder)
        # depth_model.cuda()
        # depth_model.eval()

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        img_ext = '.png' if opt.png else '.jpg'

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # output = depth_decoder(encoder(input_color))
                output = depth_model(input_color)

                # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_err", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    encoder, depth_decoder, depth_model, encoder_dict = import_model(opts, 'manydepth')

    ## evaluate attacks
    all_args = [
    {
        # 0
        "norm_type": "l_0", 
        "step": 10,
        "adam_lr": 0.5,
        "mask_wt": 0.06,
        "l0_thresh": 0.05, # 0.05 0.1 0.2 0.333
        "batch_size": 8
    }, {
        # 1
        "norm_type": "l_0", 
        "step": 10,
        "adam_lr": 0.5,
        "mask_wt": 0.06,
        "l0_thresh": 0.1, # 0.05 0.1 0.2 0.333
        "batch_size": 8
    }, 
    # {
    #     # 2
    #     "norm_type": "l_0",
    #     "step": 10,
    #     "adam_lr": 0.5,
    #     "mask_wt": 0.06,
    #     "l0_thresh": 0.2, # 0.05 0.1 0.2 0.333
    #     "batch_size": 8
    # }, {
    #     # 3
    #     "norm_type": "l_0", 
    #     "step": 10,
    #     "adam_lr": 0.5,
    #     "mask_wt": 0.06,
    #     "l0_thresh": 0.333, # 0.05 0.1 0.2 0.333
    #     "batch_size": 8
    # }, 
    {
        # 4
        "norm_type": "l_inf", 
        "epsilon": 0.05, # 0.05 0.1 0.2
        "alpha": 0.02, # 0.02 0.02 0.04
        "step": 10,
        "batch_size": 12,
    }, {
        # 5
        "norm_type": "l_inf", 
        "epsilon": 0.1, # 0.05 0.1 0.2
        "alpha": 0.02, # 0.02 0.02 0.04
        "step": 10,
        "batch_size": 12,
    },
    # {
    #     # 6
    #     "norm_type": "l_inf", 
    #     "epsilon": 0.2, # 0.05 0.1 0.2
    #     "alpha": 0.04, # 0.02 0.02 0.04
    #     "step": 10,
    #     "batch_size": 12,
    # }, {
    #     # 7
    #     "norm_type": "image", 
    #     "epsilon": 0.01, 
    #     "alpha": 0.002,
    #     "step": 10,
    #     "batch_size": 12,
    # }, 
    # {
    #     # 8
    #     "norm_type": "l_2", 
    #     "epsilon": 8, # 0.05 0.1 0.2
    #     "alpha": 0.02, # 0.02 0.02 0.04
    #     "step": 10,
    #     "batch_size": 12,
    # }, {
    #     # 9
    #     "norm_type": "l_2", 
    #     "epsilon": 16, # 0.05 0.1 0.2
    #     "alpha": 0.02, # 0.02 0.02 0.04
    #     "step": 10,
    #     "batch_size": 12,
    # },{
    #     # 10
    #     "norm_type": "l_2", 
    #     "epsilon": 24, # 0.05 0.1 0.2
    #     "alpha": 0.04, # 0.02 0.02 0.04
    #     "step": 10,
    #     "batch_size": 12,
    # },
    # {
    #     # 11
    #     "norm_type": "APGD", 
    #     "epsilon": 0.05, # 0.05 0.1 0.2
    #     "step": 10,
    #     "batch_size": 12,
    # },
    # {
    #     # 12
    #     "norm_type": "Square", 
    #     "epsilon": 0.1, # 0.05 0.1 0.2
    #     "n_queries": 5000,
    #     "batch_size": 12,
    # },
    # {
    #     # 13
    #     "norm_type": "arbi", 
    #     "batch_size": 32,
    # },
    ]

    for args in all_args:
        setup_seed(17)
        evaluate_attacks(depth_model, args)
    
    # setup_seed(17)
    # evaluate_attacks(depth_model, all_args[1], eval_count=1)



    # evaluate model performance
    evaluate(opts, encoder, depth_decoder, encoder_dict)
