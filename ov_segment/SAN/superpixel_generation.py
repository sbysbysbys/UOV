from typing import List, Union

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import argparse
import glob
import multiprocessing as mp
import os
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

import tempfile
import time

import tqdm

from tqdm import tqdm
from multiprocessing import Pool
from nuscenes.nuscenes import NuScenes

import huggingface_hub
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from huggingface_hub import hf_hub_download
from PIL import Image

from san import add_san_config
from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES
from nuscenes_tools import CategoryProcessor, metadata_nuscenes_dic

model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco.yaml",
        "model_path": "huggingface:san_vit_b_16.pth",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
        "model_path": "huggingface:san_vit_large_14.pth",
    },
}


def download_model(model_path: str):
    """
    Download the model from huggingface hub.
    Args:
        model_path (str): the model path
    Returns:
        str: the downloaded model path
    """
    if "HF_TOKEN" in os.environ:
        huggingface_hub.login(token=os.environ["HF_TOKEN"])
    model_path = model_path.split(":")[1]
    model_path = hf_hub_download("Mendel192/san", filename=model_path)
    return model_path


def setup(config_file: str, device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


class Predictor(object):
    def __init__(self, config_file: str, model_path: str, cuda: int):
        """
        Args:
            config_file (str): the config file path
            model_path (str): the model path
        """
        # device = "cuda:"+str(cuda)
        device = "cuda"
        cfg = setup(config_file, device)
        self.model = DefaultTrainer.build_model(cfg)
        if model_path.startswith("huggingface:"):
            model_path = download_model(model_path)
        print("Loading model from: ", model_path)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            model_path
        )
        print("Loaded model from: ", model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.device(device)
            self.model = self.model.cuda()

    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],
        vocabulary: List[str] = [],
        augment_vocabulary: Union[str,bool] = True,
        sp_folder: str = None,
        cuda = 0,
    ) -> Union[dict, None]:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            sp_folder (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """
        if augment_vocabulary == "CategoryProcessor":
            category_processor = CategoryProcessor()
            vocabulary = category_processor.get_stuff_classes_dic()
        else:
            vocabulary = list(set([v.lower().strip() for v in vocabulary]))
            # remove invalid vocabulary
            vocabulary = [v for v in vocabulary if v != ""]
        print("vocabulary:", len(vocabulary))
        ori_vocabulary = vocabulary
        if augment_vocabulary == "CategoryProcessor":
            a = 1
        elif isinstance(augment_vocabulary,str):
            vocabulary = self.augment_vocabulary(vocabulary, augment_vocabulary)
        else:
            vocabulary = self._merge_vocabulary(vocabulary)
        if len(ori_vocabulary) == 0:
            ori_vocabulary = vocabulary

        nuscenes_path = image_data_or_path
        assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=nuscenes_path, verbose=False
        )
        if not os.path.exists(sp_folder):
            os.makedirs(sp_folder)

        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        total_score = len(nusc.scene)
        num_segments = 4
        average_score = total_score // num_segments
        remainder = total_score % num_segments
        segments = [average_score + 1 if i < remainder else average_score for i in range(num_segments)]
        for i in range(1,num_segments):
            segments[i] += segments[i-1] 
        segments = [0] + segments
        cuda_num = cuda

        # random_select_idx = 0 
        # cycle = 1000
        text_embed_path = os.path.join(sp_folder, "text_embedding.pt")

        # for scene_idx in tqdm(range(segments[cuda_num-4], segments[cuda_num-3])):
        for scene_idx in tqdm(range(total_score)):
            scene = nusc.scene[scene_idx]
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                for camera_name in camera_list:
                    # if random_select_idx == 0:
                    cam = nusc.get("sample_data", current_sample["data"][camera_name])
                    # os.makedirs(os.path.join(sp_folder, cam["filename"].split("__")[0].split('/')[-1]), exist_ok = True)
                    # output_file = os.path.join(os.path.join(sp_folder, cam["filename"].split("__")[0].split('/')[-1]), cam["token"] + ".png")
                    output_file = os.path.join(sp_folder, cam["token"] + ".png")
                    if not os.path.exists(output_file):

                        image_data = Image.open(os.path.join(nusc.dataroot, cam["filename"]))
                        w, h = image_data.size
                        image_tensor: torch.Tensor = self._preprocess(image_data)
                        
                        with torch.no_grad():
                            result = self.model(
                                [
                                    {
                                        "image": image_tensor,
                                        "height": h,
                                        "width": w,
                                        "vocabulary": vocabulary,
                                    }
                                ]
                            )[0]
                            image_emb = result["image_emb"] 
                            text_emb = result["text_emb"] 
                            result = result["sem_seg"]
                        seg_map, mask_emb, sinfo = self._postprocess(result, image_emb, ori_vocabulary)
                        

                        if sp_folder:
                            # save
                            im = Image.fromarray(seg_map.astype(np.uint8))
                            im.save(output_file)
                            mask_emb_path = output_file.replace(".png", ".pt")
                            sinfo_path = output_file.replace(".png", ".json")
                            with open(sinfo_path,"w") as f:
                                json.dump(sinfo,f)
                                f.close()
                            torch.save(mask_emb, mask_emb_path)
                            if not os.path.exists(text_embed_path):
                                torch.save(text_emb, text_embed_path)
                            # import pdb; pdb.set_trace()
                            # output_demo_file = output_file.replace(".png", "_demo.png")
                            # self.visualize(image_data, result.argmax(dim=0).cpu().numpy(), ori_vocabulary, output_demo_file)
                current_sample_token = current_sample["next"]
                # random_select_idx += 1 
                # random_select_idx %= cycle


        return
        # return {
        #     "image": image_data,
        #     "sem_seg": seg_map,
        #     "vocabulary": ori_vocabulary,
        # }

    def visualize(
        self,
        image: Image.Image,
        sem_seg: np.ndarray,
        vocabulary: List[str],
        demo_path: str = None,
        mode: str = "overlay",
    ) -> Union[Image.Image, None]:
        """
        Visualize the segmentation result.
        Args:
            image (Image.Image): the input image
            sem_seg (np.ndarray): the segmentation result
            vocabulary (List[str]): the vocabulary used for the segmentation
            demo_path (str): the output file path
            mode (str): the visualization mode, can be "overlay" or "mask"
        Returns:
            Image.Image: the visualization result. If demo_path is not None, return None.
        """
        # add temporary metadata
        # set numpy seed to make sure the colors are the same
        # np.random.seed(0)
        # colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
        # MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)
        # metadata = MetadataCatalog.get("_temp")
        metadata = metadata_nuscenes_dic()
        if mode == "overlay":
            v = Visualizer(image, metadata)
            v = v.draw_sem_seg(sem_seg, area_threshold=0).get_image()
            v = Image.fromarray(v)
        else:
            v = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            labels, areas = np.unique(sem_seg, return_counts=True)
            sorted_idxs = np.argsort(-areas).tolist()
            labels = labels[sorted_idxs]
            for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
                v[sem_seg == label] = metadata.stuff_colors[label]
            v = Image.fromarray(v)
        # remove temporary metadata
        MetadataCatalog.remove("nuscenes_dic")
        if demo_path is None:
            return v
        v.save(demo_path)
        print(f"saved to {demo_path}")

    def _merge_vocabulary(self, vocabulary: List[str]) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        return vocabulary + [c for c in default_voc if c not in vocabulary]

    def augment_vocabulary(
        self, vocabulary: List[str], aug_set: str = "COCO-all"
    ) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        stuff_voc = [
            c["name"]
            for c in COCO_CATEGORIES
            if "isthing" not in c or c["isthing"] == 0
        ]
        if aug_set == "COCO-all":
            return vocabulary + [c for c in default_voc if c not in vocabulary]
        elif aug_set == "COCO-stuff":
            return vocabulary + [c for c in stuff_voc if c not in vocabulary]
        else:
            return vocabulary

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")
        # resize short side to 640
        w, h = image.size
        if w < h:
            image = image.resize((640, int(h * 640 / w)))
        else:
            image = image.resize((int(w * 640 / h), 640))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        image = image.permute(2, 0, 1)
        return image

    def _postprocess(
        self, result: torch.Tensor, image_emb: torch.Tensor, ori_vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=0) # (H, W)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = -1

        inds, inverse_indices = torch.unique(result, sorted=True, return_inverse=True)
        sinfo = []
        sinfo_num = 1
        # mask_emb = torch.empty((0, image_emb.shape[-1])).to(image_emb.device)
        mask_emb = torch.zeros((1, image_emb.shape[-1])).to(image_emb.device)
        for i in range(len(inds)):
            if inds[i] != -1:
                sinfo.append({})
                sinfo[-1]["id"] = sinfo_num
                sinfo_num += 1
                sinfo[-1]["category_id"] = inds[i].item()
                area = torch.where(inverse_indices[:, :]==i)
                area_size = area[0].shape[0]
                sinfo[-1]["area"] = area_size
                area_feature = F.normalize(torch.sum(image_emb[area], dim=0), dim=0)
                # area_feature = image_emb[inds[i]]
                mask_emb = torch.cat((mask_emb, area_feature.reshape(1, -1)), dim=0)
                # import pdb; pdb.set_trace()
            else:
                inverse_indices -= 1

        inverse_indices += 1
        inverse_indices = inverse_indices.cpu().numpy() 
        mask_emb = mask_emb.cpu()       
        # import pdb; pdb.set_trace()
        return inverse_indices, mask_emb, sinfo


def pre_download():
    """pre downlaod model from huggingface and open_clip to avoid network issue."""
    for model_name, model_info in model_cfg.items():
        download_model(model_info["model_path"])
        cfg = setup(model_info["config_file"])
        DefaultTrainer.build_model(cfg)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="path to config file",
        default="./configs/san_clip_vit_large_res4_coco.yaml"
    )
    parser.add_argument(
        "--model_path", type=str, help="path to model file",
        default="san_vit_large_14.pth"
    )
    parser.add_argument(
        "--root_dir", type=str, help="path to image file.",
        default='../../data/sets/nuscenes',
    )
    parser.add_argument("--aug-vocab", 
        action="store_true", 
        default="CategoryProcessor",
        help="augment vocabulary.")
    parser.add_argument(
        "--vocab",
        type=str,
        default="",
        help="list of category name. seperated with ,.",
    )
    parser.add_argument(
        "--sp_folder",
        default="../../data/SAN/superpixels",
        help="A file or directory to save sp_folder visualizations. "
        "If not given, will show sp_folder in an OpenCV window.",
    )
    parser.add_argument('-c', '--cuda', help='cuda_number', type=int,
                    default=0)
    args = parser.parse_args()
    os.makedirs(args.sp_folder, exist_ok = True)
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path, cuda=args.cuda)
    predictor.predict(
        args.root_dir,
        args.vocab.split(","),
        args.aug_vocab,
        sp_folder=args.sp_folder,
        cuda=args.cuda
    )
