import os
import copy
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from MinkowskiEngine.utils import sparse_quantize
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from utils.transforms import (
    make_transforms_clouds,
    make_transforms_asymmetrical,
    make_transforms_asymmetrical_val,
)


# parametrizing set, to try out different parameters
CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def custom_collate_fn(list_data):
    """
    Custom collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    # whether the dataset returns labels
    labelized = len(input) == 12
    # evaluation_labels are per points, labels are per voxels
    if labelized:
        (
            xyz, 
            coords, 
            feats, 
            labels, 
            evaluation_labels,
            pseudo_labels, 
            pseudo_evaluation_labels, 
            inverse_indexes,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = input
        
    else:
        # xyz, coords, feats, inverse_indexes = input
        (
            xyz, 
            coords, 
            feats, 
            inverse_indexes,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = input

    coords_batch, len_batch = [], []
    batch_n_points, batch_n_pairings = [], []

    # create a tensor of coordinates of the 3D points
    # note that in ME, batche index and point indexes are collated in the same dimension
    
    # offset = 0
    # for batch_id, coo in enumerate(coords):
    #     N = coords[batch_id].shape[0]
    #     coords_batch.append(
    #         torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
    #     )
    #     len_batch.append(N)
    # coords_batch = torch.cat(coords_batch, 0).int()

    discrete_coords = torch.cat(
        (
            torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
            discrete_coords,
        ),
        1,
    )
    offset = 0
    for batch_id in range(len(coords)):
        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        len_batch.append(coords[batch_id].shape[0])
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]
        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]
    coords_batch = torch.cat(coords, 0).int()

    # Collate all lists on their first dimension
    feats_batch = torch.cat(feats, 0).float()
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        pseudo_labels_batch = torch.cat(pseudo_labels, 0).long()
        return {
            "pc": xyz,  # point cloud
            "sinput_C": coords_batch,  # discrete coordinates (ME)
            "sinput_F": feats_batch,  # point features (N, 3)
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "pseudo_labels": pseudo_labels_batch,  # labels for each (voxelized) point
            "pseudo_evaluation_labels": pseudo_evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
            "input_I": images_batch,  # 图像
            "pairing_points": pairing_points, # 有对齐的点云
            "pairing_images": pairing_images, # 有对齐的图像
            "batch_n_pairings": batch_n_pairings, # 有对齐的点云的数量
            "superpixels": superpixels_batch, # 
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
            "input_I": images_batch,
            "pairing_points": pairing_points,
            "pairing_images": pairing_images,
            "batch_n_pairings": batch_n_pairings, 
            "superpixels": superpixels_batch,
        }


class NuScenesDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cached_nuscenes=None,
        # transforms=None,
        cloud_transforms=None,
        mixed_transforms=None,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.labels = self.phase != "test"
        # self.transforms = transforms
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"
        self.sp_dir = config["sp_dir"]
        self.task = config["task"]
        self.lidarseg_ov_dir = config["lidarseg_ov_dir"]

        if phase != "test":
            if cached_nuscenes is not None:
                self.nusc = cached_nuscenes
            else:
                self.nusc = NuScenes(
                    version="v1.0-trainval", dataroot="data/sets/nuscenes", verbose=False
                )
        else:
            self.nusc = NuScenes(
                version="v1.0-test", dataroot="data/sets/nuscenes", verbose=False
            )

        self.list_tokens = []

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase in ("val", "verifying"):
            skip_ratio = 1
        else:
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of all keyframe scenes
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_tokens(scene)

        # labels' names lookup table
        self.eval_labels = {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0, 99: 0,
        }

    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            self.list_tokens.append(current_sample["data"])
            current_sample_token = next_sample_token

    def __len__(self):
        return len(self.list_tokens)

    def __getitem__(self, idx):
        lidar_token = self.list_tokens[idx]
        # pointsensor = self.nusc.get("sample_data", lidar_token)
        # pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        # points = LidarPointCloud.from_file(pcl_path).points.T
        (
            points,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(lidar_token)
        superpixels = torch.tensor(superpixels)

        # get the points (4th coordinate is the point intensity)

        pc = torch.tensor(points[:, :3])
        intensity = torch.tensor(points[:, 3:])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # apply the transforms (augmentation)
        # if self.transforms:
        #     pc = self.transforms(pc)
        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for given voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )

        # use those voxels features
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        if self.labels:
            points_pseudo_labels = []
            unique_pseudo_labels = []
            if self.task == "zero-shot":
                _,_,lidarseg_pseudo_labels_filename = self.nusc.get("lidarseg", lidar_token)["filename"].rpartition('/')
                lidarseg_pseudo_labels_filename = os.path.join(
                    self.lidarseg_ov_dir, lidarseg_pseudo_labels_filename
                ).replace(".bin", ".npy")
                points_pseudo_labels = np.load(lidarseg_pseudo_labels_filename)%100
                points_pseudo_labels = torch.tensor(
                    np.vectorize(self.eval_labels.__getitem__)(points_pseudo_labels),
                    dtype=torch.int32,
                )
                unique_pseudo_labels = points_pseudo_labels[indexes]
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
            points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]

        if self.labels:
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                unique_pseudo_labels,
                points_pseudo_labels,
                inverse_indexes,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes
            return (
                pc,
                discrete_coords,
                unique_feats,
                inverse_indexes,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            )

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])  # ["LIDAR_TOP"]
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            # sp = Image.open(
            #     f"superpixels/nuscenes/"
            #     f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            # )
            sp = Image.open( os.path.join(self.sp_dir, f"{cam['token']}.png"))
            superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
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
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels)


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    if phase == "train":
        # transforms = make_transforms_clouds(config)
        cloud_transforms = make_transforms_clouds(config)
        mixed_transforms = make_transforms_asymmetrical(config)
    else:
        # transforms = None
        cloud_transforms = None
        mixed_transforms = make_transforms_asymmetrical_val(config)
   
    
    # instantiate the dataset
    # dset = Dataset(phase=phase, transforms=transforms, config=config)
    Dataset = NuScenesDataset
    dset = Dataset(
        phase=phase_train,  
        shuffle=phase == "train", 
        cloud_transforms=cloud_transforms,
        mixed_transforms=mixed_transforms,
        config=self.config
    )
    collate_fn = custom_collate_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
