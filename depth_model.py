import os
import sys
import torch
import torch.nn
import json
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp

class ManyDepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder, encoder_dict) -> None:
        super(ManyDepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.zero_pose = torch.zeros([1,1,4,4])
        self.encoder_dict = encoder_dict

        intrinsics_json_path=os.path.join(file_dir, 'DepthNetworks', 'manydepth2', 'assets','test_sequence_intrinsics.json')

        self.K, self.invK = load_and_preprocess_intrinsics(intrinsics_json_path,
                                             resize_width=self.encoder_dict['width'],
                                             resize_height=self.encoder_dict['height'])
        self.min_depth_bin = self.encoder_dict['min_depth_bin']
        self.max_depth_bin = self.encoder_dict['max_depth_bin']
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

    
    def forward(self, input_image):

        bs = input_image.shape[0]
        K = torch.cat(bs*[self.K])
        invK = torch.cat(bs*[self.invK])

        features, lowest_cost, _ = self.encoder(current_image=input_image,
                                         lookup_images=input_image.unsqueeze(1)*0,
                                         poses=self.zero_pose,
                                         K=K,
                                         invK=invK,
                                         min_depth_bin=self.min_depth_bin,
                                         max_depth_bin=self.max_depth_bin)

        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp/8.6437

def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def import_depth_model(scene_size, model_type='monodepth2', pre_model_path=None):
    """
    import different depth model to attack:
    possible choices: monodepth2
    """
    if scene_size == (1024, 320):
        if model_type == 'monodepth2':
            model_name = 'mono+stereo_1024x320'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'monodepth2')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        elif model_type == 'depthhints':
            model_name = 'DH_MS_320_1024'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'depth-hints')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        elif model_type == 'manydepth':
            model_name = 'KITTI_HR'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'manydepth2')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        else:
            raise RuntimeError("depth model unfound")
    else:
        raise RuntimeError("scene size undefined!")
    model_path = os.path.join(depth_model_dir, model_name)
    if pre_model_path != None:
        model_path = pre_model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    if model_type == 'manydepth':
        ## Manydepth encoder and poses network
        encoder = networks.ResnetEncoderMatching(18, False,
                                            input_width=loaded_dict_enc['width'],
                                            input_height=loaded_dict_enc['height'],
                                            adaptive_bins=True,
                                            min_depth_bin=loaded_dict_enc['min_depth_bin'],
                                            max_depth_bin=loaded_dict_enc['max_depth_bin'],
                                            depth_binning='linear',
                                            num_depth_bins=96)
    else:
        encoder = networks.ResnetEncoder(18, False)
    # encoder = networks.ResnetEncoder(18, False)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    if model_type == 'manydepth':
        depth_model = ManyDepthModelWrapper(encoder, depth_decoder, loaded_dict_enc)
    else:
        depth_model = DepthModelWrapper(encoder, depth_decoder)

    # depth_model = DepthModelWrapper(encoder, depth_decoder)
    return depth_model

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image as pil
    from torchvision import transforms
    model='manydepth'
    depth_model = import_depth_model((1024, 320), model).to(torch.device("cuda")).eval()
    img = pil.open('/home/cheng443/DepthModelHardening/asset/0000000017.png').convert('RGB')
    assert img.size == (1024, 320)
    img = transforms.ToTensor()(img).unsqueeze(0).to(torch.device("cuda"))
    img2 = img.clone().detach()
    imgs = torch.cat((img, img2), 0)
    # with torch.no_grad():
    disp = depth_model(imgs)
    print(disp.size())
    disp_np = disp.squeeze().cpu().detach().numpy()
    
    vmax = np.percentile(disp_np, 95)
    plt.figure(figsize=(5,5))
    plt.imshow(disp_np, cmap='magma', vmax=vmax)
    plt.title('Disparity')
    plt.axis('off')
    plt.savefig('temp_test.png')