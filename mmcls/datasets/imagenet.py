import os
import pdb
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class ImageNet(BaseDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    def get_class_labels(self):
        class_labels_dict = {'person': 0,
                'bicycle': 1,
                'car': 2,
                'motorcycle': 3,
                'airplane': 4,
                'bus': 5,
                'train': 6,
                'truck': 7,
                'boat': 8,
                'traffic_light': 9,
                'fire_hydrant': 10,
                'stop_sign': 11,
                'parking_meter': 12,
                'bench': 13,
                'bird': 14,
                'cat': 15,
                'dog': 16,
                'horse': 17,
                'sheep': 18,
                'cow': 19,
                'elephant': 20,
                'bear': 21,
                'zebra': 22,
                'giraffe': 23,
                'backpack': 24,
                'umbrella': 25,
                'handbag': 26,
                'tie': 27,
                'suitcase': 28,
                'frisbee': 29,
                'skis': 30,
                'snowboard': 31,
                'sports_ball': 32,
                'kite': 33,
                'baseball_bat': 34,
                'baseball_glove': 35,
                'skateboard': 36,
                'surfboard': 37,
                'tennis_racket': 38,
                'bottle': 39,
                'wine_glass': 40,
                'cup': 41,
                'fork': 42,
                'knife': 43,
                'spoon': 44,
                'bowl': 45,
                'banana': 46,
                'apple': 47,
                'sandwich': 48,
                'orange': 49,
                'broccoli': 50,
                'carrot': 51,
                'hot_dog': 52,
                'pizza': 53,
                'donut': 54,
                'cake': 55,
                'chair': 56,
                'couch': 57,
                'potted_plant': 58,
                'bed': 59,
                'dining_table': 60,
                'toilet': 61,
                'tv': 62,
                'laptop': 63,
                'mouse': 64,
                'remote': 65,
                'keyboard': 66,
                'cell_phone': 67,
                'microwave': 68,
                'oven': 69,
                'toaster': 70,
                'sink': 71,
                'refrigerator': 72,
                'book': 73,
                'clock': 74,
                'vase': 75,
                'scissors': 76,
                'teddy_bear': 77,
                'hair_drier': 78,
                'toothbrush': 79}

        return class_labels_dict

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = self.get_class_labels() 
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = self.get_class_labels()
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        idx = 0
        print(len(self.samples))
        pdb.set_trace()
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
