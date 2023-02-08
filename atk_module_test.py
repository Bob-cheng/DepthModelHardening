from physicalTrans import PhysicalTrans
from torchattacks import PGD_depth, Phy_obj_atk, Phy_obj_atk_l0
from my_utils import device0, eval_depth_diff, get_mean_depth_diff, save_pic, object_dataset_root, ori_H, ori_W
import torch
from torchvision import transforms
from depth_model import import_depth_model
from PIL import Image as pil
from image_preprocess import process_car_img
from torchvision.transforms import ToTensor
import numpy as np


scene_size = (1024, 320)
original_size = (ori_W, ori_H)
trans = transforms.Resize([int(scene_size[1]), int(scene_size[0])])

def test_adv_performance(model, image):
    atk = PGD_depth(model, eps=0.3, alpha=2/255,steps=10)
    atk._targeted = True
    adv_image, ben_image = atk(image)
    result_img, disp1, disp2 = eval_depth_diff(ben_image, adv_image, model, 'test_adv')
    mean_depth_diff = get_mean_depth_diff(torch.tensor(disp2), torch.tensor(disp1))
    print('mean_depth_diff is', mean_depth_diff)
    

def test_phy_adv_performance(model, image, obj_img, obj_mask):
    atk = Phy_obj_atk(model, obj_img, obj_mask, eps=0.3, alpha=2/255, steps=10)
    adv_scenes, ben_scenes, obj_masks_out, obj_img_adv = atk(image, 6, eval=True)
    for i in range(adv_scenes.size()[0]):
        save_pic(adv_scenes[i], f'temp_advscene_{i}')
    result_img, disp1, disp2 = eval_depth_diff(ben_scenes[[0]], adv_scenes[[0]], model, 'test_phy_adv')
    obj_masks_out = obj_masks_out.cpu()
    mean_depth_diff = get_mean_depth_diff(torch.tensor(disp2), torch.tensor(disp1), scene_car_mask=obj_masks_out)
    print('mean_depth_diff is', mean_depth_diff)

def test_phy_adv_performance_l0(model, image, obj_img, obj_mask):
    values = []
    for _ in range(10):
        atk = Phy_obj_atk_l0(model, obj_img, obj_mask, adam_lr=0.5, steps=10, mask_wt=0.06) # may need more test
        adv_scenes, ben_scenes, obj_masks_out, obj_img_adv = atk(image, 12, eval=True)
        for i in range(adv_scenes.size()[0]):
            save_pic(adv_scenes[i], f'temp_advscene_l0_{i}')
        result_img, disp1, disp2 = eval_depth_diff(ben_scenes[[0]], adv_scenes[[0]], model, 'test_phy_l0_adv')
        obj_masks_out = obj_masks_out.cpu()
        mean_depth_diff = get_mean_depth_diff(torch.tensor(disp2), torch.tensor(disp1), scene_car_mask=obj_masks_out)
        values.append(mean_depth_diff)
        print('mean_depth_diff is', np.mean(mean_depth_diff))
    print('Overall mean_depth_diff is', np.mean(values))


if __name__ == '__main__':
    model = import_depth_model(scene_size).to(device0)
    scene_img = pil.open(f'{object_dataset_root}/training/image_2/000042.png').convert('RGB')
    assert scene_img.size == (ori_W, ori_H)
    scene_img = transforms.ToTensor()(scene_img).unsqueeze(0).to(device0)
    
    test_adv_performance(model, scene_img)

    car_img_resize, car_mask_np, _ = process_car_img('BMW.png', '-2')
    obj_tensor = ToTensor()(car_img_resize)[:3,:,:].unsqueeze(0).float().to(device0)
    mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).unsqueeze(0).float().to(device0)

    test_phy_adv_performance(model, scene_img, obj_tensor, mask_tensor)

    test_phy_adv_performance_l0(model, scene_img, obj_tensor, mask_tensor)