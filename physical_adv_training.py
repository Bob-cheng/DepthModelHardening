from matplotlib.pyplot import get
import torch
import os
from PIL import Image as pil
from torch.utils import data
from depth_model import import_depth_model
from torchvision import transforms
from torchattacks.attacks import phy_obj_atk
from my_utils import device0, eval_depth_diff, get_mean_depth_diff
from torchattacks import PGD_depth, Phy_obj_atk
from torch.utils.data.dataloader import DataLoader
from dataLoader import KittiLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import datetime
from image_preprocess import process_car_img
from torchvision.transforms import ToTensor
from my_utils import save_pic, ori_H, ori_W



atk_eps = 0.03
atk_alpha = 2/255
atk_step = 10
scene_size = (1024, 320)
original_size = (ori_W, ori_H)
trans = transforms.Resize([int(scene_size[1]), int(scene_size[0])])

log_dir = os.path.join('/data/cheng443/model_harden', 'logs', datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)


def test_adv_performance(model, image):
    atk = PGD_depth(model, eps=atk_eps, alpha=atk_alpha,steps=atk_step)
    atk._targeted = True
    adv_image = atk(image)
    result_img, disp1, disp2 = eval_depth_diff(image, adv_image, model, 'test_adv')
    mean_depth_diff = get_mean_depth_diff(torch.tensor(disp2), torch.tensor(disp1))
    print('mean_depth_diff is', mean_depth_diff)
    
    

def eval_atk_perf(model_gt, model, data_loader):
    depth_atk = PGD_depth(model, eps=atk_eps, alpha=atk_alpha,steps=atk_step)
    depth_atk._targeted = True
    model.eval()
    model_gt.eval()
    model_acc, atk_perf = 0, 0
    eval_count = 100
    for  i, (scene_img, _) in enumerate(data_loader):
        if i == eval_count:
            break
        scene_img = scene_img.to(device0)
        scene_img = trans(scene_img)
        adv_image = depth_atk(scene_img)
        with torch.no_grad():
            disp_gt = model_gt(scene_img)
            disp_pre = model(scene_img)
            disp_atk = model(adv_image)
        model_acc += get_mean_depth_diff(disp_pre, disp_gt, use_abs=True) # the lower, the better model performance
        atk_perf += get_mean_depth_diff(disp_atk, disp_gt, use_abs=True) # the higher, the better attack, the lower robustness
    # data_size = len(data_loader)
    return model_acc / eval_count, atk_perf / eval_count

def do_adv_training(model_rob, model):
    model_ori = model
    model_ori.eval()
    
    total_epoch = 20
    batch_size = 6
    kitti_loader_train = KittiLoader(mode='train',  train_list='trainval.txt', val_list='test.txt', size=original_size)
    train_loader = DataLoader(kitti_loader_train, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    kitti_loader_test = KittiLoader(mode='val',  train_list='trainval.txt', val_list='test.txt', size=original_size)
    test_loader = DataLoader(kitti_loader_test, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    print(f"Train set size: {len(kitti_loader_train)}, Test set size: {len(kitti_loader_test)}")

    optimizer = optim.Adam(model_rob.parameters(), lr=0.0001)
    depth_atk = PGD_depth(model_rob, eps=atk_eps, alpha=atk_alpha,steps=atk_step)
    depth_atk._targeted = True
    loss_creteria = torch.nn.MSELoss()

    model_perf, atk_perf = eval_atk_perf(model_ori, model_rob, test_loader)
    logger.add_scalar('/eval/model_perf', model_perf, 0)
    logger.add_scalar('/eval/atk_perf', atk_perf, 0)
    print(f"Initial performance: model perf: {model_perf}, attack perf: {atk_perf}")

    for epoch in range(total_epoch):
        print('Current epoch: ', epoch)
        # train
        model_rob.train()
        for i, (scene_img, _) in enumerate(train_loader):
            scene_img = scene_img.to(device0)
            scene_img = trans(scene_img)
            with torch.no_grad():
                disp_gt = model_ori(scene_img)
            
            adv_image = depth_atk(scene_img)
            pre_disp = model_rob(adv_image)
            loss = loss_creteria(disp_gt, pre_disp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print("Current step: ", i)

        # evaluate
        model_rob.eval()
        model_perf, atk_perf = eval_atk_perf(model_ori, model_rob, test_loader)
        logger.add_scalar('/eval/model_perf', model_perf, epoch+1)
        logger.add_scalar('/eval/atk_perf', atk_perf, epoch+1)    
        print(f"Performance: model perf: {model_perf}, attack perf: {atk_perf}")


if __name__ == "__main__":
    # model = import_depth_model(scene_size).to(device0)
    # model_rob = import_depth_model(scene_size).to(device0)
    # img = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/gen_img/scene/0000000017.png').convert('RGB')
    # assert img.size == (1024, 320)
    # img = transforms.ToTensor()(img).unsqueeze(0).to(device0)
    # test_adv_performance(model, img)
    # do_adv_training(model_rob, model)

    
    scene_size = (1024, 320)
    model = import_depth_model(scene_size).to(device0)
    car_img_resize, car_mask_np, _ = process_car_img('BMW.png', '-2')
    # print('img size: ', car_img_resize.size, 'mask size: ', car_mask_np.shape)
    img_tensor = ToTensor()(car_img_resize)[:3,:,:].unsqueeze(0).float().to(device0)
    mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).unsqueeze(0).float().to(device0)
    # print('img tensor size: ', img_tensor.size(), 'mask tensor size: ', mask_tensor.size())
    scene_img = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/src_img/scene/0000000009.png').convert('RGB')
    assert scene_img.size == original_size
    scene_img = ToTensor()(scene_img).unsqueeze(0).to(device0)
    atk = Phy_obj_atk(model, img_tensor, mask_tensor, batch_size=3, steps=10)
    cfg_path = '/data/cheng443/kitti/object/training/calib/003086.txt'
    adv_scenes = atk(scene_img, cfg_path)
    for i in range(adv_scenes.size()[0]):
        save_pic(adv_scenes[i], f'temp_advscene_{i}')
