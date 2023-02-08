import os
import PIL.Image as pil
from PIL import ImageOps
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

src_car_path = os.path.join(dir_path, 'asset', 'src_img', 'object')
gen_car_path     = os.path.join(dir_path, 'asset', 'gen_img', 'object')

car_img_width = 300
scene_size = (1024, 320) # width, height

def prepare_dir():
    if not os.path.exists(gen_car_path):
        os.makedirs(gen_car_path)

def process_img(img_name, output_w, image_type: str):
    if image_type == 'car':
        img_path = os.path.join(src_car_path, img_name)
        img_out_path = os.path.join(gen_car_path, img_name)
    if not os.path.exists(img_path):
        raise RuntimeError("image '%s' doesn't exist" % img_path)
    style_img = pil.open(img_path)
    original_w, original_h = style_img.size
    print("Image original size (w, h): (%d, %d)" % (original_w, original_h))

    output_h = int(output_w / original_w * original_h)
    style_img_resize = style_img.resize((output_w, output_h))
    style_img_resize.save(img_out_path)
    print("Output image size", style_img_resize.size)
    return style_img_resize, output_w, output_h

def process_mask(mask_name, output_w, output_h, image_type: str):
    if image_type == 'car':
        mask_path = os.path.join(src_car_path, mask_name)
        mask_out_path = os.path.join(gen_car_path, mask_name)
    if not os.path.exists(mask_path):
        img_mask_np = np.ones((output_h, output_w), dtype=int)
        print(f"The mask [{mask_name}] doesn't exist, using the whole image...")
    else:
        img_mask = ImageOps.grayscale(pil.open(mask_path))
        img_mask_np = np.array(img_mask.resize((output_w, output_h)))/255.0
        img_mask_np[img_mask_np > 0.5] = 1
        img_mask_np[img_mask_np <= 0.5] = 0
        img_mask_np = img_mask_np.astype(int)
    pil.fromarray((img_mask_np*255).astype(np.uint8), 'L').save(mask_out_path)
    return img_mask_np


def process_car_img(img_name, paintMask_no : str, mask_step: int = 1):
    ext_split = os.path.splitext(img_name)
    car_img_resize, w, h = process_img(img_name, car_img_width, 'car')
    car_mask_np = process_mask(ext_split[0] + '_CarMask' + ext_split[1], w, h, 'car')
    # if paintMask_no == '-1' or paintMask_no == '-2' : # half mask
    if int(paintMask_no) < 0: # half mask
        mask_shape = [ (i // mask_step) for i in car_mask_np.shape ]
        # paint_mask_np = np.random.random(mask_shape)
        paint_mask_np = np.ones(mask_shape) * 0.5
        paint_mask_np =  np.clip(paint_mask_np, 0.0, 1.0)
    else:
        paint_mask_np = process_mask(ext_split[0] + '_PaintMask' + paintMask_no + ext_split[1], w, h, 'car')
    print(ext_split[0] + '_PaintMask' + paintMask_no + ext_split[1])
    assert car_img_resize.size[::-1] == car_mask_np.shape
    return car_img_resize, car_mask_np, paint_mask_np


if __name__ == '__main__':
    prepare_dir()
    process_car_img("BMW.png", paintMask_no='-2')
    