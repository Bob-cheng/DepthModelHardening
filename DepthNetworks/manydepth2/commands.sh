#! /bin/bash

 python evaluate_depth.py --load_weights_folder /data3/user/cheng443/model_harden/tmp/training/manyd_stereo_advTrain_l0_v2_f0adv_0aug_Ori_halfADV_bencomp/models/weights_2 --width 1024 --height 320 --eval_mono --data_path /data3/share/kitti/kitti_data/ --png

 python evaluate_depth.py --load_weights_folder /home/cheng443/DepthModelHardening/DepthNetworks/manydepth2/models/KITTI_HR --width 1024 --height 320 --eval_mono --data_path /data3/share/kitti/kitti_data/ --png
