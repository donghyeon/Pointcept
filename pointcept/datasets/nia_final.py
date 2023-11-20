from collections.abc import Sequence
from pathlib import Path
import concurrent.futures

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import json
import polars as pl

from .builder import DATASETS
from .defaults import DefaultDataset

from .nia_dataset_reader import (
    NiaDataPathExtractor,
    DataFrameSplitter,
    NiaDataPathProvider,
)

def find_common_points(cloud1, cloud2):
    # Create a cKDTree object for each point cloud
    tree1 = cKDTree(cloud1)
    tree2 = cKDTree(cloud2)
    
    # Fast search indices of common points of two point clouds
    # ex) indices: [[tree2_index_i], [tree2_index_j], [], ...]
    # len(indices) == len(tree1)
    common_point_indices = tree1.query_ball_tree(tree2, r=0)
    
    # overlapping_indices = []
    # for i, indices in enumerate(common_point_indices):
    #     for j in indices:
    #         overlapping_indices.append((i, j))
    
    return common_point_indices


def find_common_points_in_tuple(clouds):
    cloud1, cloud2 = clouds
    return find_common_points(cloud1, cloud2)


def split_by_yanghoon(sequence):
    val_set = set(filter(lambda sample: '230822/230822_162106_K' in sample[0], sequence))
    train_set = set(sequence) - val_set
    test_set = []
    
    return [list(train_set), list(val_set), test_set]


def split_by_ratio(sequence, ratio, random_seed=None):
    length = len(sequence)
    if random_seed is not None:
        import random
        nia_random = random.Random(random_seed)
        nia_random.shuffle(sequence)
    normalized_cumsum = np.cumsum(ratio) / np.sum(ratio)
    indices = list((normalized_cumsum * length).astype(int))[:-1]
    return [
        sequence[start:end]
        for start, end in zip([None] + indices, indices + [None])
    ]


# TODO: Figure out why remove_duplicated_points() occurs segmentation fault when num_workers > 0
def read_lidar_pcd(lidar_path):
    cloud = o3d.t.io.read_point_cloud(lidar_path)
    # cloud = cloud.remove_duplicated_points()[0]

    feature_keys: list[str] = dir(cloud.point)
    coord = cloud.point["positions"].numpy()

    if "reflectivity" in feature_keys:
        strength_key = "reflectivity"
    elif "intensity" in feature_keys:
        strength_key = "intensity"
    else:
        raise Exception(f"No strength key in {lidar_path}")
    strength = cloud.point[strength_key].numpy() / 255
    # return coord, strength

    # Filter out same points
    unique_coord, unique_indices = np.unique(coord, return_index=True, axis=0)

    return unique_coord, strength[unique_indices]


def read_json(json_path):
    with open(json_path) as f:
        label = json.load(f)
    return label


# TODO: Create segment by read_annotations and match_annotations (annotation["3d_points"] and coord)
def read_annotations(label):
    annotations = label["annotations"]
    return list(map(lambda x: (x["3d_points"], x["class_id"]), annotations))


@DATASETS.register_module()
class NiaFinalDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="/datasets/nia/",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.path_provider = NiaDataPathProvider(
            NiaDataPathExtractor(
                dataset_dir=data_root,
                pattern=(
                    r"(?P<type>[^/]+)/"
                    r"(?P<collector>[^/]+)/"
                    r".*?"
                    r"(?P<channel>[^/]+)/"
                    r"(?P<filename>[^/]+)$"
                ),
                exclude_filenames=["LV_B03_R01_night_clear_01091948.pcd"],  # File corrupted
            ),
            DataFrameSplitter(
                groups=["channel", "collector", "code_1", "code_2", "timeslot", "weather"],
                splits=["train", "valid", "test"],
                ratios=[8, 1, 1],
                seed=231111,
            ),
            channels=["lidar"],
        )
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def create_segment_array(self, label_path, cloud_points):
        label = read_json(label_path)

        segment = np.ones((cloud_points.shape[0],), dtype=int) * self.ignore_index

        annotations = label['annotations']
        for annotation in annotations:
            instance_points = annotation['3D_points']
            instance_class_id = annotation['class_id']

            indices = find_common_points(instance_points, cloud_points)
            indices = np.array(indices, dtype=int).reshape(-1)
            segment[indices] = instance_class_id
        
        return segment
    
    def create_segment_array_in_parallel(self, label_path, cloud_points):
        label = read_json(label_path)

        annotations = label['annotations']

        all_instance_points = map(lambda x: x['3D_points'], annotations)
        all_instance_class_ids = map(lambda x: x['class_id'], annotations)

        num_workers = 8  # The number of threads or processes
        with concurrent.futures.ProcessPoolExecutor(num_workers) as mp_executor:
            all_instance_indices = mp_executor.map(
                find_common_points_in_tuple,
                zip(all_instance_points, [cloud_points] * len(annotations)),
            )

        segment = np.ones((cloud_points.shape[0],), dtype=int) * self.ignore_index
        for indices, class_id in zip(all_instance_indices, all_instance_class_ids):
            indices = np.array(indices, dtype=int).reshape(-1)
            segment[indices] = class_id
        
        return segment
    
    def get_split_data_list(self, split, small_valid_num_samples=None):
        split_data_list = self.path_provider.get_split_data_list(split)
        if split == "valid":
            if small_valid_num_samples is not None:
                import random
                nia_random = random.Random(small_valid_num_samples)
                nia_random.shuffle(split_data_list)
            split_data_list = split_data_list[:small_valid_num_samples]
        
        return split_data_list
    
    def get_data_list(self):
        data_list = []
        if isinstance(self.split, str):
            data_list += self.get_split_data_list(self.split)
        elif isinstance(self.split, Sequence):
            for s in self.split:
                data_list += self.get_split_data_list(s)
        else:
            raise NotImplementedError
        return data_list
    
    def get_data(self, idx, parallel=False):
        lidar_path, label_path = self.data_list[idx % len(self.data_list)]

        # Read lidar
        coord, strength = read_lidar_pcd(lidar_path)

        # Read label
        if parallel:
            segment = self.create_segment_array_in_parallel(label_path, coord)
        else:
            segment = self.create_segment_array(label_path, coord)
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
            np.int64
        )

        # Filter out ignore_index
        ignore_points_indices = segment != self.ignore_index
        data_dict = dict(
            coord=coord[ignore_points_indices],
            strength=strength[ignore_points_indices],
            segment=segment[ignore_points_indices],
        )
        # data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        lidar_path = self.data_list[idx % len(self.data_list)][0]
        data_name = Path(lidar_path).name
        return data_name
    
    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            ignore_index: ignore_index,
            # 0: 2,  # "Two-wheel Vehicle",  # Class index error fixed in final data
            1: 3,  # "Pedestrian"
            2: 2,  # "Two-wheel Vehicle",
            3: 0,  # "Car"
            8: 1,  # "Truck/Bus"
            10: 7,  # "Traffic Light"
            12: 6,  # "Traffic Sign"
            40: 4,  # "Road"
            48: 5,  # "Sidewalk"
        }
        return learning_map
