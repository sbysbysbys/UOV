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
from torch import nn
from torch.nn import functional as F
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

from cat_seg import add_cat_seg_config
from predictor import VisualizationDemo
from nuscenes_tools import metadata_nuscenes_dic, metadata_nuscenes_2, metadata_nuscenes


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.SEM_SEG_HEAD.SP_FOLDER = args.sp_folder
    cfg.freeze()
    return cfg

def compute_catseg(cam, output_filename, text_embed_path, text_len):
    img = read_image(os.path.join(nusc.dataroot, cam["filename"]), format="BGR")

    masks = demo.predictor(img)
    # masks, visualized_output = demo.run_on_image(img)
    # visualized_output.save(output_filename.replace('.png', '_demo.png'))

    result = masks["sem_seg"].argmax(dim=0).detach()
    image_emb = masks["image_embed"].permute(1,2,0)
    text_embed = masks["text_embed"]
    if not os.path.exists(text_embed_path):
        torch.save(text_embed, text_embed_path)

    result[result >= text_len] = -1
    inds, inverse_indices = torch.unique(result, sorted=True, return_inverse=True)
    sinfo = []
    sinfo_num = 1
    mask_emb = torch.zeros((1, image_emb.shape[-1])).to(image_emb.device)
    # import pdb; pdb.set_trace()
    for i in range(len(inds)):
        if inds[i] != -1:
            sinfo.append({})
            sinfo[-1]["id"] = sinfo_num
            sinfo_num += 1
            sinfo[-1]["category_id"] = inds[i].item()
            area = torch.where(inverse_indices[:, :]==i)
            area_size = area[0].shape[0]
            sinfo[-1]["area"] = area_size
            # area_feature = torch.sum(image_emb[area], dim=0)/area_size
            area_feature = F.normalize(torch.sum(image_emb[area], dim=0), dim=0)
            mask_emb = torch.cat((mask_emb, area_feature.reshape(1, -1)), dim=0)
        else:
            inverse_indices -= 1
    inverse_indices += 1
    segment_catseg = inverse_indices.cpu().numpy() 
    mask_emb = mask_emb.cpu()

    im = Image.fromarray(segment_catseg.astype(np.uint8))
    im.save(output_filename)
    mask_emb_path = output_filename.replace(".png", ".pt")
    sinfo_path = output_filename.replace(".png", ".json")
    with open(sinfo_path,"w") as f:
        json.dump(sinfo,f)
        f.close()
    torch.save(mask_emb, mask_emb_path)

    # import pdb; pdb.set_trace()

def get_parser():
    parser = argparse.ArgumentParser(description="catseg demo for builtin configs")
    parser.add_argument(
        "--config_file",
        default="configs/nuscenes.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--root_folder",
        default='../../data/sets/nuscenes',
    )
    parser.add_argument(
        "--sp_folder",
        default="../../data/CAT-Seg/superpixels",
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
        default=['MODEL.WEIGHTS', 'model_final_large.pth'],
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
    metadata = metadata_nuscenes_dic()
    text_len = len(metadata.stuff_classes)

    random_select_idx = 0 
    cycle = 1

    # with open("nuscenes_pics.txt", "a")as f:
    # for scene_idx in tqdm(range(segments[cuda_num-3], segments[cuda_num-2])):
    for scene_idx in tqdm(range(total_score)):
        scene = nusc.scene[scene_idx]
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            for camera_name in camera_list:
                if random_select_idx == 0:
                    cam = nusc.get("sample_data", current_sample["data"][camera_name])
                    output_file = os.path.join(args.sp_folder, cam["token"] + ".png")
                    # f.write(os.path.join(nusc.dataroot, cam["filename"])+"\\\\"+output_file+"\n")
                    if not os.path.exists(output_file):
                        compute_catseg(cam, output_file, text_embed_path, text_len)
            random_select_idx += 1 
            random_select_idx %= cycle
            current_sample_token = current_sample["next"]


