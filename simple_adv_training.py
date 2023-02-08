import torch
import os
from depth_model import import_depth_model
from torchvision import transforms
from my_utils import device0, eval_depth_diff, get_mean_depth_diff, ori_H, ori_W, object_dataset_root, save_depth_model
from torchattacks import PGD_depth, Phy_obj_atk, Phy_obj_atk_l0
from torch.utils.data.dataloader import DataLoader
from dataLoader import KittiLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import datetime 
import numpy as np
import random
from options import getCLIOptions
from image_preprocess import process_car_img

args = getCLIOptions()
print(args)

scene_size = (1024, 320)
original_size = (ori_W, ori_H)
trans = transforms.Resize([int(scene_size[1]), int(scene_size[0])])

# log_dir = os.path.join('/data/cheng443/model_harden', 'logs', datetime.datetime.now().strftime('%b%d_%H-%M-%S') + f"_{args['log_postfix']}")
# log_dir = os.path.join('/home/jcl3689/zhiyuan/tmp/model_harden_log', datetime.datetime.now().strftime('%b%d_%H-%M-%S') + f"_{args['log_postfix']}")
log_dir = os.path.join('/home/jcl3689/zhiyuan/tmp/model_harden_log', f"{args['log_postfix']}")
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)
logger.add_text('Options', str(args), 0)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_atk_model(model_rob):
    if args['adv_type'] == 'image':
        depth_atk = PGD_depth(model_rob, eps=args['epsilon'], alpha=args['alpha'],steps=args['step'])
        depth_atk._targeted = True
    else:
        car_img_resize, car_mask_np, _ = process_car_img('BMW.png', '-2')
        obj_tensor = transforms.ToTensor()(car_img_resize)[:3,:,:].unsqueeze(0).float().to(device0)
        mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).unsqueeze(0).float().to(device0)
        if args['adv_type'] == 'object':
            depth_atk = Phy_obj_atk(model_rob, obj_tensor, mask_tensor,\
                eps=args['epsilon'], alpha=args['alpha'], steps=args['step'])
        elif args['adv_type'] == 'object_l0':
            depth_atk = Phy_obj_atk_l0(model_rob, obj_tensor, mask_tensor, \
                adam_lr=args['adam_lr'], steps=args['step'], mask_wt=args['mask_wt'], l0_thresh=args['l0_thresh'])
        else: 
            raise NotImplementedError('adv_type not implemented')
        # logger.add_image('input/obj_img', obj_tensor[0], 0)
        # logger.add_image('input/obj_mask', mask_tensor[0], 0)
    return depth_atk


def eval_atk_perf(model_gt, model, data_loader, epoch):
    depth_atk = get_atk_model(model)
    model.eval()
    model_gt.eval()
    model_acc, atk_perf = 0, 0
    eval_count = 50
    start_idx = 42
    for  i, (scene_img, _) in enumerate(data_loader):
        if i < start_idx:
            continue
        if i - start_idx >= eval_count:
            break
        scene_img = scene_img.to(device0)
        # scene_img = trans(scene_img)
        if args['adv_type'] == 'object' or args['adv_type'] == 'object_l0':
            adv_images, ben_images, obj_masks_out, obj_img_adv = depth_atk(scene_img, args['batch_size'], eval=True)
        elif args['adv_type'] == 'image':
            adv_images, ben_images = depth_atk(scene_img)
            obj_masks_out = None
        with torch.no_grad():
            disp_gt = model_gt(ben_images)
            disp_pre = model(ben_images)
            disp_atk = model(adv_images)
        if i == start_idx:
            result_img_model, _, _ = eval_depth_diff(ben_images[[0]], ben_images[[0]], model, disp1=disp_pre[[0]], disp2=disp_gt[[0]])
            result_img_atk, _, _ = eval_depth_diff(adv_images[[0]], ben_images[[0]], model_gt, disp1=disp_atk[[0]], disp2=disp_gt[[0]])
        model_acc += get_mean_depth_diff(disp_pre, disp_gt, scene_car_mask=None, use_abs=True) # the lower, the better model performance
        atk_perf += get_mean_depth_diff(disp_atk, disp_gt, scene_car_mask=obj_masks_out, use_abs=True) # the higher, the better attack, the lower robustness
    # data_size = len(data_loader)
    logger.add_image('eval/model_comp', transforms.ToTensor()(result_img_model), epoch)
    logger.add_image('eval/atk_comp', transforms.ToTensor()(result_img_atk), epoch)
    model_perf = model_acc / eval_count
    atk_perf = atk_perf / eval_count
    logger.add_scalar('eval/model_perf', model_perf, epoch)
    logger.add_scalar('eval/atk_perf', atk_perf, epoch)
    return model_perf, atk_perf

def do_adv_training(model_rob, model):
    model_ori = model
    model_ori.eval()
    
    total_epoch = 20
    # if args['adv_type'] == 'object':
    #     loader_batch_size = 1
    # elif args['adv_type'] == 'image':
    #     loader_batch_size = args['batch_size']
    loader_batch_size = args['batch_size']

    kitti_loader_train = KittiLoader(mode='train', root_dir=object_dataset_root, train_list='trainval.txt', val_list='test.txt', size=original_size)
    train_loader = DataLoader(kitti_loader_train, batch_size=loader_batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)

    kitti_loader_test = KittiLoader(mode='val', root_dir=object_dataset_root, train_list='trainval.txt', val_list='test.txt', size=original_size)
    test_loader = DataLoader(kitti_loader_test, batch_size=loader_batch_size, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)

    print(f"Train set size: {len(kitti_loader_train)}, Test set size: {len(kitti_loader_test)}")

    optimizer = optim.Adam(model_rob.parameters(), lr=0.0001)
    depth_atk = get_atk_model(model_rob)
    loss_creteria = torch.nn.MSELoss()

    model_perf, atk_perf = eval_atk_perf(model_ori, model_rob, test_loader, 0)
    print(f"Initial performance: model perf: {model_perf}, attack perf: {atk_perf}")
    
    for epoch in range(total_epoch):
        print('Current epoch: ', epoch)
        # train
        model_rob.train()
        for i, (scene_img_ori, _) in enumerate(train_loader):
            scene_img_ori = scene_img_ori.to(device0)
            if args['adv_type'] == 'object' or args['adv_type'] == 'object_l0':
                adv_images, ben_images, obj_masks_out, obj_img_adv = depth_atk(scene_img_ori, args['batch_size'])
            elif args['adv_type'] == 'image':
                adv_images, ben_images = depth_atk(scene_img_ori)
            
            with torch.no_grad():
                disp_gt = model_ori(ben_images)
            
            pre_disp = model_rob(adv_images)
            loss = loss_creteria(disp_gt, pre_disp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print("Current step: ", i)

        # evaluate
        model_rob.eval()
        model_perf, atk_perf = eval_atk_perf(model_ori, model_rob, test_loader, epoch+1)
        print(f"Performance: model perf: {model_perf}, attack perf: {atk_perf}")
        # save model
        if epoch % 2 == 0:
            save_depth_model(model_rob, log_dir, epoch+1)
            # torch.save(model_rob.state_dict(), os.path.join(log_dir, f'model_rob_ep_{epoch+1}.pt'))
    # end training
    save_depth_model(model_rob, log_dir, 'final')
    # torch.save(model_rob.state_dict(), os.path.join(log_dir, f'model_rob_ep_final.pt'))


if __name__ == "__main__":
    setup_seed(args['random_seed'])
    model = import_depth_model(scene_size).to(device0)
    model_rob = import_depth_model(scene_size).to(device0)
    # torch.save(model_rob.state_dict(), os.path.join(log_dir, f'model_rob_ep_test.pt'))
    do_adv_training(model_rob, model)


