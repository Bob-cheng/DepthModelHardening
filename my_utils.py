import torch
from PIL import Image as pil
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torchvision import transforms
from depth_model import DepthModelWrapper
import os

device0 = torch.device("cuda")
object_dataset_root = "/data3/share/kitti/object/"
ori_H = 375
ori_W = 1242
train_dist_range = list(np.arange(5, 10, 0.2)) 
# train_dist_range = [5] * 30
# train_dist_range = list(np.arange(5, 31, 2)) 

# [5] * 30 / list(np.arange(5, 15, 0.2)) /  list(np.arange(5, 20, 0.2)) ...

def disp_to_depth(disp,min_depth,max_depth):
# """Convert network's sigmoid output into depth prediction
# The formula for this conversion is given in the 'additional considerations'
# section of the paper.
# """
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth

def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask=None, use_abs=False):
    scaler=5.4
    if scene_car_mask == None:
        scene_car_mask = torch.ones_like(adv_disp1)
    dep1_adv=torch.clamp(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask*scaler,max=100)
    dep2_ben=torch.clamp(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask*scaler,max=100)
    if use_abs:
        mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask)
    else:
        mean_depth_diff = torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_car_mask)
    return mean_depth_diff

def eval_depth_diff(img1: torch.tensor, img2: torch.tensor, depth_model, filename=None, disp1=None, disp2=None):
    if disp1 == None:
        disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    else:
        disp1 = disp1.detach().cpu().squeeze().numpy()
    if disp2 == None:
        disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    else:
        disp2 = disp2.detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(321); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(323)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    plt.subplot(324)
    plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(325)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    plt.subplot(326)
    plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    if filename != None:
        plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image, disp1, disp2

def eval_depth_diff_jcl(img1: torch.tensor, img2: torch.tensor, depth_model, filename=None, disp1=None, disp2=None):
    if disp1 == None:
        disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    else:
        disp1 = disp1.detach().cpu().squeeze().numpy()
    if disp2 == None:
        disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    else:
        disp2 = disp2.detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(311); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    # plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(312)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    # plt.subplot(324)
    # plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(313)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    # plt.subplot(326)
    # plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    if filename != None:
        plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image

def save_depth_model(model: DepthModelWrapper, log_folder, epoch, width=1024, height=320, use_stereo=True):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_folder, "models", "weights_{}".format(epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, "{}.pth".format('encoder'))
    to_save = model.encoder.state_dict()
    # save the sizes - these are needed at prediction time
    to_save['height'] = height
    to_save['width'] = width
    to_save['use_stereo'] = use_stereo
    torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format('depth'))
    to_save = model.decoder.state_dict()
    torch.save(to_save, save_path)



def save_pic(tensor, i, log_dir=''):
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if log_dir != '':
        file_path = os.path.join(log_dir, "{}.png".format(i))
    else:
        file_path = "{}.png".format(i)
    image.save(file_path, "PNG")