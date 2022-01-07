import json
import csv
import os
import random
import numpy as np
from typing import Callable
from typing import List, Dict
from typing import Optional
from typing import Tuple

import torch
import pytorch_lightning as pl
import torch
import random
import numpy
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torchvision.transforms import transforms

from slot_attention.utils import compact
from clevr_obj_test.test_generation import obj_algebra_test, attr_algebra_test


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        data_weights: Dict = None,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = data_root
        self.max_n_objects = max_n_objects
        self.data_weights = data_weights
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        image_path = self.files[index]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        paths = []
        for dataset, weight in self.data_weights.items():
            print(dataset, weight)
            if not np.isnan(weight):
                data_path = os.path.join(self.data_root, dataset, "images")
                assert os.path.exists(data_path), f"Path {data_path} does not exist"
                image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
                paths += image_paths[:int(self.max_num_images*weight)]
        ##paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if os.path.isfile(os.path.join(self.data_root, f))]
        # with open(os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")) as f:
        #     scene = json.load(f)
        # paths: List[Optional[str]] = []
        # total_num_images = len(scene["scenes"])
        # i = 0
        # while (self.max_num_images is None or len(paths) < self.max_num_images) and i < total_num_images:
        #     num_objects_in_scene = len(scene["scenes"][i]["objects"])
        #     if num_objects_in_scene <= self.max_n_objects:
        #         image_path = os.path.join(self.data_path, scene["scenes"][i]["image_filename"])
        #         assert os.path.exists(image_path), f"{image_path} does not exist"
        #         paths.append(image_path)
        #     i += 1
        return sorted(compact(paths[:self.max_num_images]))

class CLEVRValset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        val_list: List[List[Optional[str]]] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_main_scenes = max_num_images
        self.img_root =  os.path.join(data_root, "images")
        self.mask_root = os.path.join(data_root, "masks")
        self.meta_root = os.path.join(data_root, "meta")
        self.max_n_objects = max_n_objects

        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert os.path.exists(self.img_root), f"Path {self.img_root} does not exist"
        assert os.path.exists(self.mask_root), f"Path {self.mask_root} does not exist"
        assert os.path.exists(self.meta_root), f"Path {self.meta_root} does not exist"

        if val_list:
            self.img_files = val_list
        else:
            self.img_files = self.get_files()

    def __getitem__(self, index: int):
        image_paths = self.img_files[index]
        imgs = [Image.open(image_path) for image_path in image_paths[:-1]]
        imgs = [img.convert("RGB") for img in imgs]
        meta = np.load(image_paths[-1])
        schema = np.concatenate([meta['size'], meta['material'], meta['shape'], meta['color'], meta['x'], meta['y'], meta['z'], meta['rotation']], axis=0)
        schema = torch.from_numpy(schema).permute(1,0)
        return [self.clevr_transforms(img) for img in imgs]+[schema]

    def __len__(self):
        return len(self.img_files)

    def get_files(self) -> List[str]:
        paths: List[List[Optional[str]]] = []
        i = 0
        while (self.max_num_main_scenes is None or i < self.max_num_main_scenes) and i < 100000:
            meta_path = os.path.join(self.meta_root, '{}.npz'.format(i))
            meta = np.load(meta_path, allow_pickle=True)

            num_objects_in_scene = int(meta['visibility'].sum())
            if num_objects_in_scene <= self.max_n_objects:
                img_paths = []
                image_path = os.path.join(self.img_root, '{}.png'.format(i))
                assert os.path.exists(image_path), f"{image_path} does not exist"
                img_paths.append(image_path)
                for j in range(1, self.max_n_objects+2): # For FG ARI
                    mask_path = os.path.join(self.mask_root, '{}_{}.png'.format(i, j))
                    assert os.path.exists(mask_path), f"{mask_path} does not exist"
                    img_paths.append(mask_path)
                paths.append(img_paths+[meta_path])
            i += 1
        with open(os.path.join(self.data_root, "CLEVR_val_list.csv"), "w") as f:
            wr = csv.writer(f)
            wr.writerows(paths)
        return paths

class CLEVRAlgebraTestset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        test_type: str = "obj", # or "attr"
        test_cases: List[List[Optional[str]]] = None,
        num_test_cases: int = 50000,
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_main_scenes = max_num_images
        if test_type == "obj":
            self.test_root =  os.path.join(data_root, "obj_test_occ")
        else:
            raise NotImplemented
        self.data_path = os.path.join(self.test_root, "images")
        self.max_n_objects = max_n_objects
        self.test_type = test_type
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.test_type == "obj" or self.test_type == "attr"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        if test_cases:
            self.img_files = test_cases
        else:
            self.img_files = self.get_files()
        if len(self.img_files) > num_test_cases:
            self.img_files = self.img_files[0:num_test_cases]
        print("Test case size: ", len(self.img_files))

    def __getitem__(self, index: int):
        image_paths = self.img_files[index]
        imgs = [Image.open(image_path) for image_path in image_paths]
        imgs = [img.convert("RGB") for img in imgs]
        return [self.clevr_transforms(img) for img in imgs]

    def __len__(self):
        return len(self.img_files)

    def get_files(self) -> List[str]:
        print("creating file path for "+f"{self.test_type}_test")
        with open(os.path.join(self.test_root, "CLEVR_scenes.json")) as f:
            scene = json.load(f)
        paths: List[List[Optional[str]]] = []
        total_num_main_scenes = len(scene["scenes"])
        i = 0
        while (self.max_num_main_scenes is None or i < self.max_num_main_scenes) and i < total_num_main_scenes:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            # if num_objects_in_scene <= self.max_n_objects:
            # First, call obj_algebra_test or attr_algebra_test with this scene to generate path tuples for A-B+C=D
            if self.test_type == 'obj':
                image_paths = obj_algebra_test(self.test_root, i)
            elif self.test_type == 'attr':
                image_paths = attr_algebra_test(self.test_root, i)
            else:
                raise NotImplementedError
            # Then, assert the existence of these paths
            for image_path in image_paths:
                for path in image_path:
                    assert os.path.exists(path), f"{path} does not exist"
            # Last, append these path tuples into paths.
            paths+=image_paths
            i += 1
        random.shuffle(paths)
        with open(os.path.join(self.test_root, "CLEVR_test_cases.csv"), "w") as f:
            wr = csv.writer(f)
            wr.writerows(paths)
        return paths


class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        val_root: str,
        test_root: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        clevr_transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
        num_test_images: Optional[int] = None,
        data_weights: Dict=None,
        val_list: List[List[Optional[str]]] = None,
        obj_algebra_test_cases: List[List[Optional[str]]] = None,
        attr_algebra_test_cases: List[List[Optional[str]]] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.val_root = val_root
        self.test_root = test_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.clevr_transforms = clevr_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images
        self.num_test_images = num_test_images

        self.train_dataset = CLEVRDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
            data_weights=data_weights,
        )
        self.val_dataset = CLEVRValset(
            data_root=self.val_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            max_n_objects=self.max_n_objects,
            val_list = val_list,
        )

        self.obj_test_dataset = CLEVRAlgebraTestset(
            data_root = self.test_root,
            max_num_images=self.num_test_images,
            clevr_transforms = self.clevr_transforms,
            max_n_objects = self.max_n_objects,
            test_type = "obj",
            test_cases = obj_algebra_test_cases,
        )

    def train_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def val_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def obj_test_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        rand_sampler = RandomSampler(self.obj_test_dataset)
        return DataLoader(
            self.obj_test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=rand_sampler,
            worker_init_fn=seed_worker,
            generator=g,
        )


class CLEVRTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
