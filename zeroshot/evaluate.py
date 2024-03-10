import os
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
from utils.metrics import compute_IoU
# import sys
# sys.path.append('/root/wangfeiyue3new/sby/post_process')
# from post_process import get_point_nn, post_process


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


def evaluate(model, dataloader, config, save=False):
    """
    Function to evaluate the performances of a zeroshot training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []

        # device = "cuda:"+str(cuda)
        # if config["post_process"] :
        #     point_nn = get_point_nn(config, device)

        # for batch in tqdm(dataloader):
        for batch in dataloader:
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=0)
            output_points = model(sparse_input).F
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # # postprocess
                # if config["post_process"] :
                #     pc = batch["pc"][j].unsqueeze(0).float().to(device)
                #     ov_seg = (batch["pseudo_evaluation_labels"][j]%100).unsqueeze(0).long().to(device)
                #     output_seg = (predictions%100).unsqueeze(0).long().to(device)
                #     # import pdb; pdb.set_trace()
                #     point_labels, _ = post_process(point_nn, config, pc, output_seg, ov_seg)
                #     predictions = point_labels.squeeze(0).cpu()

                if save and i%100 == 0: 
                    lidarseg_filename = batch["lidarseg_filename"][j].replace(".bin", ".npy")
                    # print(lidarseg_filename)
                    pc = batch["pc"][j]
                    pseudo_superpoints = batch["pseudo_evaluation_labels"][j]
                    gt_superpoints = batch["evaluation_labels"][j]
                    save_path = config["pp_save_path"]
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, lidarseg_filename)
                    combined = np.hstack((np.array(pc), np.array(predictions.reshape(-1,1))))
                    combined = np.hstack((combined, np.array(pseudo_superpoints.reshape(-1,1))))
                    combined = np.hstack((combined, np.array(gt_superpoints.reshape(-1,1))))
                    np.save(save_path, combined)

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
                i += 1
        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            torch.cat(full_predictions),
            torch.cat(ground_truth),
            config["model_n_out"],
            ignore_index=0,
        )
        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
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

    return m_IoU
