import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import argparse
from pointwidth_features import Point_NN_Seg
from nuscenes_tools import CategoryProcessor
from sklearn.linear_model import RANSACRegressor, LinearRegression
base_estimator = LinearRegression()
ransac = RANSACRegressor(estimator=base_estimator, min_samples=10, residual_threshold=0.1, max_trials=20, random_state=42)
from copy import deepcopy
import sys
sys.path.append('..')
from utils.metrics import compute_IoU

CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

def generate_text_embedding(text_features_path, device):
    category_processor = CategoryProcessor()
    trans_a = category_processor.get_stuff_dataset_id_to_class_id()
    trans_b = category_processor.get_class_id_to_real_class_id()
    trans_c = category_processor.get_real_class_id_to_category_id()

    stuff_embed = torch.load(text_features_path)[:-1].to(device)

    stuff_to_class = np.vectorize(trans_a.__getitem__)(np.arange(len(trans_a)))
    stuff_to_real_class = np.vectorize(trans_b.__getitem__)(stuff_to_class)
    stuff_to_category = np.vectorize(trans_c.__getitem__)(stuff_to_real_class)
    stuff_to_category = torch.tensor(stuff_to_category).to(device)
    values = torch.unique(stuff_to_category, sorted=True, return_inverse=False, return_counts=False)

    text_embed = torch.zeros((values.shape[0], stuff_embed.shape[1]))
    for category in values[1:]:
        mask = stuff_to_category == category
        category_rows = stuff_embed[mask]
        # import pdb; pdb.set_trace()
        category_sum = F.normalize(torch.sum(category_rows, dim=0) / category_rows.shape[0], dim=0)
        text_embed[category] = category_sum
    return text_embed.to(device)

def post_process(point_nn, cfg, pc, output_seg = None, ov_seg = None, text_embed = None):
    d = torch.sqrt(pc[:, :, 0]**2 + pc[:, :, 1]**2)
    
    del_mask = torch.ones(d.shape).to(pc.device)
    if cfg["dataset"] == "nuscenes":
        # 去除周围地面
        round_aera = (d < cfg["del_max_distance"]) & (d > 2) & (pc[:, :, 2] < -1.4)
        round_aera_idx = torch.where(round_aera)
        pc_round_aera = pc[round_aera_idx].cpu()
        # import pdb; pdb.set_trace()
        # RANSACRegressor
        ransac.fit(pc_round_aera[:,0:2], pc_round_aera[:,2])
        inlier_mask = torch.tensor(ransac.inlier_mask_).to(pc.device)
        inlier_mask_idx = torch.where(inlier_mask)
        round_ground_idx = round_aera_idx[1][inlier_mask_idx]
        del_mask[:,round_ground_idx] = 0
        del_mask = del_mask.bool() & (d > 2)
    del_mask = del_mask.unsqueeze(-1)  # .repeat(1,1,cfg["dim"])
    

    # merge output and ov labels
    if ov_seg is not None:
        beta = cfg["beta"]
        samep_dist = cfg["samep_dist"]
        temperature = samep_dist / math.log(beta)
        p = beta * torch.exp(-d / temperature) / (1 + beta * torch.exp(-d / temperature))
        p = p * (ov_seg != 0).to(p.device) # .double().squeeze()
        mask = torch.rand(p.shape).to(p.device) < p
        mask *= (ov_seg < 11)
        seg = torch.where(mask, ov_seg, output_seg)
    elif ov_seg is None:
        seg = output_seg
    elif output_seg is None:
        seg = ov_seg 

    # label_features = text_embed[seg]
    only_hot_embd = torch.eye(cfg["dim"]).to(seg.device)
    label_features = only_hot_embd[seg]
    # replace_mask = ~(del_mask.repeat(1,1,cfg["dim"]-1)) * (1/(cfg["dim"]-1))
    # replace_mask = torch.cat((torch.zeros(del_mask.shape).to(del_mask.device), replace_mask), dim=-1)
    label_features = del_mask * label_features   # + replace_mask
    point_features = point_nn(pc.permute(0, 2, 1), label_features) #fast
    if cfg["dataset"] == "nuscenes":
        ground_mask = torch.zeros(point_features.shape[0], 1, point_features.shape[2]).to(point_features.device)
        ground_mask[:,:,11:15] = 1
        point_features[:, round_ground_idx] *= ground_mask
    # import pdb; pdb.set_trace()
    _, point_labels = torch.max(point_features, dim=-1)
    p_changed = torch.sum(point_labels != seg) / point_labels.shape[-1]
    return point_labels, p_changed

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/semseg_nuscenes_zeroshot.yaml")
    parser.add_argument('--sp_folder', type=str, default="data/SLidr/fcclip")
    parser.add_argument('--post_process', type=str)
    parser.add_argument('--dataset', type=str)  # 71.27, 73.95

    parser.add_argument('--pp_batch_size', type=int)  # Freeze as 1

    parser.add_argument('--points', type=int)
    parser.add_argument('--stages', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--de_k', type=int)  # propagate neighbors in decoder
    parser.add_argument('--alpha', type=int)
    parser.add_argument('--beta2', type=int)
    parser.add_argument('--gamma', type=int)  # Best as 300
    parser.add_argument('--delta', type=int)
    parser.add_argument('--cuda', type=int)

    args = parser.parse_args()
    return args

def get_point_nn(cfg, device):
    point_nn = Point_NN_Seg(input_points=cfg["points"], num_stages=cfg["stages"],
                            embed_dim=cfg["dim"], k_neighbors=cfg["k"], de_neighbors=cfg["de_k"],
                            alpha=cfg["alpha"], beta=cfg["beta2"], gamma=cfg["gamma"], delta=cfg["delta"]).to(device)
    point_nn.eval()
    return point_nn

if __name__ == "__main__":
    args = get_arguments()
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        if arg_value is not None:
            cfg[str(arg_name)] = arg_value
    # print(cfg)
    point_nn = get_point_nn(cfg, "cuda:"+str(cfg["cuda"]))

    # generate text embedding
    # text_embed = generate_text_embedding(cfg["text_features_path"], "cuda:"+str(cfg["cuda"]))

    full_predictions = []
    ground_truth = []
    for root, dirs, files in os.walk(args.sp_folder):
        # for name in tqdm(files):
        for name in files:
            lidar_path = os.path.join(root, name)
            # print(lidar_path)
            pointcloud = torch.tensor(np.load(lidar_path)).unsqueeze(0).to("cuda:"+str(cfg["cuda"]))
            pc = pointcloud[:, :, :3].float()
            output_seg = (pointcloud[:, :, 3]%100).long()
            ov_seg = (pointcloud[:, :, 4]%100).long()
            gt_seg = (pointcloud[:, :, 5]%100).long()
            
            # point_labels, p_changed = post_process(point_nn, cfg, pc, output_seg, ov_seg, text_embed)

            ground_truth.append(gt_seg.squeeze(0).cpu())
            if cfg["post_process"]:
                # change here
                if cfg["if_ovseg"]:
                    point_labels, p_changed = post_process(point_nn, cfg, pc, output_seg, ov_seg)
                else:
                    point_labels, p_changed = post_process(point_nn, cfg, pc, output_seg)
                full_predictions.append(point_labels.squeeze(0).cpu())
            else:
                full_predictions.append(output_seg.squeeze(0).cpu())

    m_IoU, fw_IoU, per_class_IoU = compute_IoU(
        torch.cat(full_predictions),
        torch.cat(ground_truth),
        cfg["model_n_out"],
        ignore_index=0,
    )
    print("Per class IoU:")
    if cfg["dataset"].lower() == "nuscenes":
        print(
            *[
                f"{a:20} - {b:.3f}"
                for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
            ],
            sep="\n",
        )
    elif cfg["dataset"].lower() == "kitti":
        print(
            *[
                f"{a:20} - {b:.3f}"
                for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
            ],
            sep="\n",
        )
    print()
    print(f"mIoU: {m_IoU}")
    print(f"fwIoU: {fw_IoU}")


