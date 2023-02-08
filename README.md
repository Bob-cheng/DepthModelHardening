# Depth Model Hardening


This is the official PyTorch implementation of our paper *Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks* accepted at ICLR23 (Spotlight). 




## Table of Contents

1. [Requirements and Installation](#Requirements-and-Installation)
2. [Getting Started](#Getting-Started)




## Requirements and Installation


### :hammer: Installation

```
conda create --name depthhardening --file requirements.txt
```

### 💾 KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i ./DepthNetworks/monodepth2/splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

Another dataset we used in this project is kitti object detection dataset. It can be downloaded [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Then you need to organize the data in the following way. The image split files can be downloaded [here](https://github.com/charlesq34/frustum-pointnets/tree/master/kitti/image_sets). 
```
KITTI/object/
    
    train.txt
    val.txt
    test.txt 
    
    training/
        calib/
        image_2/ #left image
        image_3/ #right image
        label_2/
        velodyne/ 

    testing/
        calib/
        image_2/
        image_3/
        velodyne/
```
After downloading the dataset, please set the variable `object_dataset_root` in file `./my_utils.py` to the path to the object dataset.

### Original MDE Models:

Download the original [Monodepth2](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip) and [DepthHints](https://storage.googleapis.com/niantic-lon-static/research/depth-hints/DH-HR-Mono%2BStereo/models.zip) models and put the model files under the following folders. Make sure at least a `depth.pth` and a `encoder.pth` is inside the folder.
```
# Original Monodepth2 Model
./DepthNetworks/monodepth2/models/mono+stereo_1024x320/

# Original DepthHints Model
./DepthNetworks/depth-hints/models/DH_MS_320_1024/
```



## Getting Started

### Adversairal Model Training
Here is an example of model hardening training for monodepth2 with self-supervised training, contrastive learning and supervised learning.

```shell
cd ./DepthModelHardening/DepthNetworks/monodepth2

python train.py \
--model_name depth_hardening \ 
--frame_ids 0 \
--use_stereo \
--split eigen_full \
--png --data_path /path/to/kitti/data_set/ \
--width 1024 \
--height 320 \
--learning_rate 0.00001 \
--log_dir ~/path/to/log/dir/ \
--adv_train \
--norm_type l_0 \
--num_workers 8 \
--batch_size 32 \
--contrastive_learning \
--supervised_adv \
```
### Defensive Performance Evaluation

Run the following to evaluate the defensive performance of a given model.

```shell
cd ./DepthModelHardening/DepthNetworks/monodepth2

python evaluate_depth.py \ 
--load_weights_folder /path/to/log/dir/model_name/weights_2 \
--width 1024 \
--height 320 \
--eval_stereo \
--data_path /data3/share/kitti/kitti_data/ 
--png
```

To see the meaning of each option, use `--help` flag. You can also refer the [Monodepth2 project](https://github.com/nianticlabs/monodepth2) for explanation.

## Citation

Please cite our paper at:
```
@inproceedings{
    cheng2023adversarial,
    title={Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks},
    author={Zhiyuan Cheng and James Chenhao Liang and Guanhong Tao and Dongfang Liu and Xiangyu Zhang},
    booktitle={International Conference on Learning Representations},
    year={2023}
}
```
