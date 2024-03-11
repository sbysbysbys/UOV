# modified from SLidR (https://github.com/valeoai/SLidR/blob/main/pretrain/dataloader_nuscenes.py)

import os
import numpy as np
import torch
import copy
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import pickle
from demo.nuscenes_tools import CategoryProcessor
seed = 1242
random.seed(seed)
np.random.seed(seed)

eval_labels = {
        0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
        12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0, 99: 99,
    }
def map_pointcloud_to_image(data, sp_root, args, min_dist: float = 1.0, if_save = False):
    """
    Given a lidar token and camera sample_data token, load pointcloud and map it to
    the image plane. Code adapted from nuscenes-devkit
    https://github.com/nutonomy/nuscenes-devkit.
    :param min_dist: Distance from the camera below which points are discarded.
    """
    category_processor = CategoryProcessor()
    get_class_id = category_processor.get_stuff_dataset_id_to_class_id()
    get_real_class_id = category_processor.get_class_id_to_real_class_id()
    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    pointsensor = nusc.get("sample_data", data["LIDAR_TOP"])
    pcl_path = os.path.join(nusc.dataroot, pointsensor["filename"])
    lidarseg_path = os.path.join(nusc.dataroot, nusc.get("lidarseg", data["LIDAR_TOP"])["filename"])
    pc_original = LidarPointCloud.from_file(pcl_path)
    pc_ref = pc_original.points

    label_set = np.ones((len(pc_ref.T)),dtype=np.int32) * -1  # (m,)
    label = 0  
    categories = []
    features = []
    
    for i, camera_name in enumerate(camera_list):
        pc = copy.deepcopy(pc_original)
        cam = nusc.get("sample_data", data[camera_name])
        im = np.array(Image.open(os.path.join(nusc.dataroot, cam["filename"])))
        sp_path = sp_root +"/" + cam["token"] + ".png"
        sinfo_path = sp_root +"/" + cam["token"] + ".json"
        rf_path = sp_root +"/" + cam["token"] + ".pt"
        sp = Image.open(sp_path)
        sp = np.array(sp)

        # Points live in the point sensor frame. So they need to be transformed via
        # global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the
        # timestamp of the sweep.
        cs_record = nusc.get(
            "calibrated_sensor", pointsensor["calibrated_sensor_token"]
        )
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        # Third step: transform from global into the ego vehicle frame for the
        # timestamp of the image.
        poserecord = nusc.get("ego_pose", cam["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get(
            "calibrated_sensor", cam["calibrated_sensor_token"]
        )
        pc.translate(-np.array(cs_record["translation"]))
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Take the actual picture
        # (matrix multiplication with camera-matrix + renormalization).
        points = view_points(
            pc.points[:3, :],
            np.array(cs_record["camera_intrinsic"]),
            normalize=True,
        )

        # Remove points that are either outside or behind the camera.
        # Also make sure points are at least 1m in front of the camera to avoid
        # seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        points = points[:2].T
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[:, 0] > 0)
        mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
        mask = np.logical_and(mask, points[:, 1] > 0)
        mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
        matching_points = np.where(mask)[0]
        matching_pixels = np.round(
            np.flip(points[matching_points], axis=1)
        ).astype(np.int64)

        sp_value = sp[matching_pixels[:, 0], matching_pixels[:, 1]]  # (m_match,)
        inds, counts = np.unique(sp_value, return_counts=True)

        with open(sinfo_path, 'r') as file:
            sinfo = json.load(file)
            file.close()
        # import pdb; pdb.set_trace()
        for ii in inds:
            if ii == 0:
                continue
            # region features
            mask_info = sinfo[ii-1]
            mask_category = mask_info["category_id"]
            mask_org_len =  np.zeros(mask.shape[0], dtype=bool) 
            mask2 = np.in1d(sp_value, ii)
            mask_org_len[mask] = mask2
            label_set[mask_org_len] = label*100 + get_real_class_id[get_class_id[mask_category]] # 少一个noise
            label += 1
            

    assert len(pc_ref.T) == len(label_set)

    # psame
    lidarseg_dir = args.lidarseg_save_folder
    points_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
    p_sameseg = np.sum(np.equal((label_set % 100), points_labels))/len(label_set)
    p_labeled = len([x for x in label_set if x != -1])/len(label_set)
    p_sameseg = p_sameseg/p_labeled

    
    if if_save:
        _,_,lidarseg_filename = lidarseg_path.rpartition('/')
        lidarseg_save_path = os.path.join(lidarseg_dir, lidarseg_filename)
        lidarseg_save_path = lidarseg_save_path.replace(".bin", ".npy")
        np.save(lidarseg_save_path, label_set)
        # feature_save_path = lidarseg_save_path.replace(".npy", ".npz")
        # np.savez(feature_save_path, *features)
        # import pdb; pdb.set_trace()
        # combined = np.hstack((np.array(pc_ref.T), np.array(label_set.reshape(-1,1))))
        # np.save(lidarseg_save_path, combined)
        

    return p_sameseg, p_labeled
    # return pc_ref.T, label_set.reshape(-1,1)


def create_list_of_tokens(scene):
    # Get first in the scene
    current_sample_token = scene["first_sample_token"]

    # Loop to get all successive keyframes
    while current_sample_token != "":
        current_sample = nusc.get("sample", current_sample_token)
        next_sample_token = current_sample["next"]
        list_tokens.append(current_sample["data"])
        current_sample_token = next_sample_token



def parse_arguments():

    parser = argparse.ArgumentParser(description='superpixel2superpoint')
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='/root/wangfeiyue3new/sby/Segment-Any-Point-Cloud/data/sets/nuscenes')
    parser.add_argument('-s', '--sp_folder', help='superpixels root', type=str,
                        default='../../data/CAT-Seg/superpixels') 
    parser.add_argument('-l', '--lidarseg_save_folder', help='save lidarseg', type=str,
                        default='../../data/CAT-Seg/lidarseg_ov') 
    arguments = parser.parse_args()

    return arguments



if __name__ == "__main__":
    args = parse_arguments()
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=args.root_folder, verbose=False)
    # phase_scenes = create_splits_scenes()['train']
    phase_scenes = create_splits_scenes()['train'] + create_splits_scenes()['val']
    skip_counter = 0
    skip_ratio = 1
    list_tokens = []
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]
        if scene["name"] in phase_scenes:
            skip_counter += 1
            if skip_counter % skip_ratio == 0:
                create_list_of_tokens(scene)
    print("finish sence_idx")

    os.makedirs(args.lidarseg_save_folder, exist_ok = True)

    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]
    ps = []
    pl = []
    for idx in tqdm(range(len(list_tokens))):
        lidar_token = list_tokens[idx]
        all_check = 1
        for i, camera_name in enumerate(camera_list):
            cam = nusc.get("sample_data", lidar_token[camera_name])
            sp_path = args.sp_folder +"/" + cam["token"] + ".png"
            if not os.path.exists(sp_path):
                all_check = 0  
        lidarseg_path = os.path.join(nusc.dataroot, nusc.get("lidarseg", lidar_token["LIDAR_TOP"])["filename"])
        _,_,lidarseg_filename = lidarseg_path.rpartition('/')
        lidarseg_save_path = os.path.join(args.lidarseg_save_folder, lidarseg_filename)
        lidarseg_save_path = lidarseg_save_path.replace(".bin", ".npy")
        if os.path.exists(lidarseg_save_path):
            all_check = 0
        if all_check:
            p_sameseg, p_labeled = map_pointcloud_to_image(lidar_token, args.sp_folder, args = args, if_save = True)


            