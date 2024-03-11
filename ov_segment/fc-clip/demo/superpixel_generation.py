"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse
import glob
import multiprocessing as mp
import os
import json


# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import torch
import matplotlib.pyplot as plt

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from nuscenes.nuscenes import NuScenes

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from fcclip import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def compute_fcclip(cam, output_filename, text_embed_path):
    img = read_image(os.path.join(nusc.dataroot, cam["filename"]), format="BGR")

    masks = demo.predictor(img)
    # masks, visualized_output = demo.run_on_image(img)
    # visualized_output.save(output_filename.replace('.png', '_demo.png'))

    panoptic_seg, segments_info, fin, fout = masks["panoptic_seg"]
    f_in_out = [fin.detach().cpu(), fout.detach().cpu()]
    torch.save(f_in_out, output_filename.replace('png', 'pt'))

    text_embed = masks["text_embedding"]
    if not os.path.exists(text_embed_path):
        torch.save(text_embed, text_embed_path)

    segment_fcclip = panoptic_seg.detach().cpu().numpy()

    im = Image.fromarray(segment_fcclip.astype(np.uint8))
    im.save(output_filename)
    
    with open(output_filename.replace('png', 'json'),"w") as f:
        json.dump(segments_info,f)
        f.close()

def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_nuscenes.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--root_folder",
        default='../../data/sets/nuscenes',
    )
    parser.add_argument(
        "--sp_folder",
        default="../../data/fc-clip/superpixels",
        help="A file or directory to save sp_folder visualizations. "
        "If not given, will show sp_folder in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'fcclip_cocopan.pth'],
        # default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('-c', '--cuda', help='cuda_number', type=int,
                        default=0)
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    nuscenes_path = args.root_folder
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"

    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=False
    )
    if not os.path.exists(args.sp_folder):
        os.makedirs(args.sp_folder)

    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    total_score = len(nusc.scene)
    num_segments = 1
    average_score = total_score // num_segments
    remainder = total_score % num_segments
    segments = [average_score + 1 if i < remainder else average_score for i in range(num_segments)]
    for i in range(1,num_segments):
        segments[i] += segments[i-1] 
    segments = [0] + segments
    cuda_num = args.cuda

    text_embed_path = os.path.join(args.sp_folder, "text_embedding.pt")

    # for scene_idx in tqdm(range(segments[cuda_num], segments[cuda_num+1])):
    for scene_idx in tqdm(range(total_score)):
        scene = nusc.scene[scene_idx]
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            for camera_name in camera_list:
                cam = nusc.get("sample_data", current_sample["data"][camera_name])
                # os.makedirs(os.path.join(args.sp_folder, cam["filename"].split("__")[0].split('/')[-1]), exist_ok = True)
                # output_file = os.path.join(os.path.join(args.sp_folder, cam["filename"].split("__")[0].split('/')[-1]), cam["token"] + ".png")
                output_file = os.path.join(args.sp_folder, cam["token"] + ".png")
                if not os.path.exists(output_file):
                    compute_fcclip(cam, output_file, text_embed_path)

            current_sample_token = current_sample["next"]


