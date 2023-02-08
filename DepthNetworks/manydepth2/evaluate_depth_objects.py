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
from depth_model import DepthModelWrapper
from torchattacks.attacks.phy_obj_atk import Phy_obj_atk
from torchattacks.attacks.phy_obj_atk_l0 import Phy_obj_atk_l0
from torchattacks.attacks.pgd_depth import PGD_depth
from image_preprocess import process_car_img
from dataLoader import KittiLoader
from my_utils import object_dataset_root, ori_W, ori_H, eval_depth_diff, get_mean_depth_diff

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

def evaluate_attacks_obj(object_name: str, model2atk: DepthModelWrapper, args, eval_count=25):
    print(args)
    car_img_resize, car_mask_np, _ = process_car_img(object_name, '-2')
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

    kitti_loader_test = KittiLoader(mode='val', root_dir=object_dataset_root, train_list='trainval.txt', val_list='test.txt', size=(ori_W, ori_H))
    test_loader = DataLoader(kitti_loader_test, batch_size=args['batch_size'], shuffle=False, num_workers=args['batch_size'], pin_memory=True, drop_last=True)
    atk_perf = 0
    start_idx = 42
    errors = []
    for  i, (scene_img, _) in enumerate(test_loader):
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
    

    # result_img_atk.save('./temp_atk_perform.png')
    # atk_perf = atk_perf / eval_count
    # print("attack performance: mean depth estimation eror: ", atk_perf)

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_err", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Attack Eval Done!")
    return mean_errors


def import_model(model_folder, layers=18):
    encoder_path = os.path.join(model_folder, "encoder.pth")
    decoder_path = os.path.join(model_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    encoder = networks.ResnetEncoder(layers, False)
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



if __name__ == "__main__":
    # options = MonodepthOptions()
    # opts = options.parse()
    models_folder = "/home/cheng443/data/model_harden/tmp/training/{}"

    candi_models = [
        "mono_original",
        "stereo_advTrain_l0_v2_f0adv_0aug_lre-5/models/weights_2",
        "stereo_advTrain_l0_v2_f0adv_0aug_ContrasOnly_lre-5/models/weights_2",
        "stereo_advTrain_l0_v2_f0adv_0aug_SupOnly_lre-5/models/weights_2"
    ]

    candi_obj = "TrafficBarrier2.png"
    # BMW.png / Sedan_Back.png / Subaru.png / Truck_Back.png / SUV_Back.png / TrafficBarrier2.png

    atk_args = {
        # 1
        "norm_type": "l_0", 
        "step": 10,
        "adam_lr": 0.5,
        "mask_wt": 0.06,
        "l0_thresh": 0.1, # 0.05 0.1 0.2 0.333
        "batch_size": 8
    }

    for i in range(len(candi_models)):
        source_folder = models_folder.format(candi_models[i])
        _, _, source_model, _ = import_model(source_folder)
        setup_seed(17)
        evaluate_attacks_obj(candi_obj, source_model, atk_args)
