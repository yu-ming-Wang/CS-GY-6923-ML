# import torch
# from torch.utils.data import DataLoader
# from dpcv.data.transforms.transform import set_vat_transform_op
# from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData
# from dpcv.data.datasets.tpn_data import TPNData as VATData
# from dpcv.data.datasets.tpn_data import TPNTruePerData as VATTruePerData
# from dpcv.data.datasets.tpn_data import FullTestTPNData as FullTestVATData
# from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop, TemporalDownsample, TemporalEvenCropDownsample
# from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
# from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
# from dpcv.data.transforms.build import build_transform_spatial
# from dpcv.data.datasets.common import VideoLoader


# def make_data_loader(cfg, mode="train"):

#     assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = set_vat_transform_op()
#     temporal_transform = [TemporalRandomCrop(16)]
#     temporal_transform = TemporalCompose(temporal_transform)
#     video_loader = VideoLoader()

#     if mode == "train":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TRAIN_IMG_DATA,
#             cfg.TRAIN_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "valid":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.VALID_IMG_DATA,
#             cfg.VALID_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "trainval":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TRAINVAL_IMG_DATA,
#             cfg.TRAINVAL_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     else:
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TEST_IMG_DATA,
#             cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=cfg.TRAIN_BATCH_SIZE,
#         shuffle=cfg.SHUFFLE,
#         num_workers=cfg.NUM_WORKERS,
#     )
#     return data_loader


# @DATA_LOADER_REGISTRY.register()
# def vat_data_loader(cfg, mode="train"):

#     assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
#         "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = build_transform_spatial(cfg)
#     temporal_transform = [TemporalDownsample(length=100), TemporalRandomCrop(16)]
#     # temporal_transform = [TemporalDownsample(length=16)]
#     temporal_transform = TemporalCompose(temporal_transform)

#     data_cfg = cfg.DATA
#     if "face" in data_cfg.TRAIN_IMG_DATA:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
#     else:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

#     if mode == "train":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TRAIN_IMG_DATA,
#             data_cfg.TRAIN_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "valid":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.VALID_IMG_DATA,
#             data_cfg.VALID_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "trainval":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TRAINVAL_IMG_DATA,
#             data_cfg.TRAINVAL_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "full_test":
#         temporal_transform = [TemporalDownsample(length=100), TemporalEvenCropDownsample(16, 6)]
#         temporal_transform = TemporalCompose(temporal_transform)
#         return FullTestVATData(
#             data_cfg.ROOT,
#             data_cfg.TEST_IMG_DATA,
#             data_cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     else:
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TEST_IMG_DATA,
#             data_cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )

#     loader_cfg = cfg.DATA_LOADER
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=loader_cfg.TRAIN_BATCH_SIZE,
#         shuffle=loader_cfg.SHUFFLE,
#         num_workers=loader_cfg.NUM_WORKERS,
#     )
#     return data_loader


# @DATA_LOADER_REGISTRY.register()
# def true_per_vat_data_loader(cfg, mode="train"):
#     assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
#         "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = build_transform_spatial(cfg)
#     temporal_transform = [TemporalRandomCrop(16)]
#     # temporal_transform = [TemporalDownsample(length=2000), TemporalRandomCrop(16)]
#     temporal_transform = TemporalCompose(temporal_transform)

#     data_cfg = cfg.DATA
#     if data_cfg.TYPE == "face":
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
#     else:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

#     data_set = VATTruePerData(
#         data_root="datasets/chalearn2021",
#         data_split=mode,
#         task=data_cfg.SESSION,
#         data_type=data_cfg.TYPE,
#         video_loader=video_loader,
#         spa_trans=spatial_transform,
#         tem_trans=temporal_transform,
#     )

#     shuffle = True if mode == "train" else False
#     loader_cfg = cfg.DATA_LOADER
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=loader_cfg.TRAIN_BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=loader_cfg.NUM_WORKERS,
#     )
#     return data_loader


# if __name__ == "__main__":
#     import os
#     from dpcv.config.tpn_cfg import cfg

#     os.chdir("../../")

#     # interpret_data = InterpretData(
#     #     data_root="datasets",
#     #     img_dir="image_data/valid_data",
#     #     label_file="annotation/annotation_validation.pkl",
#     #     trans=set_transform_op(),
#     # )
#     # print(interpret_data[18])

#     data_loader = make_data_loader(cfg, mode="valid")
#     for i, item in enumerate(data_loader):
#         print(item["image"].shape, item["label"].shape)
#         if i > 5:
#             break

# ---------------------------------------- 改过之后 ---------------------------------------------
# import torch
# from torch.utils.data import DataLoader
# from dpcv.data.transforms.transform import set_vat_transform_op
# from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData
# from dpcv.data.datasets.tpn_data import TPNData as VATData
# from dpcv.data.datasets.tpn_data import TPNTruePerData as VATTruePerData
# from dpcv.data.datasets.tpn_data import FullTestTPNData as FullTestVATData
# from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop, TemporalDownsample, TemporalEvenCropDownsample
# from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
# from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
# from dpcv.data.transforms.build import build_transform_spatial
# from dpcv.data.datasets.common import VideoLoader

# import glob
# import zipfile
# import tempfile
# import shutil
# import os
# import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# def unzip_files(zip_path):
#     temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(temp_dir)
#     return temp_dir

# class VATData(VATData):
#     def __init__(self, data_root, img_dir, label_file, video_loader, spatial_transform=None, temporal_transform=None):
#         super().__init__(data_root, img_dir, label_file, video_loader, spatial_transform, temporal_transform)
#         self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
#         self.temp_dirs = []
#         logging.debug(f"Detected {len(self.zip_files)} zip files in {img_dir}")

#     def __getitem__(self, index):
        
#         # 这个index是用来干嘛的？ 这个index有问题
#         # 这个img_dir_ls 是用来干嘛的？
#         temp_dir = unzip_files(self.zip_files[index])
#         self.temp_dirs.append(temp_dir)
#         logging.debug(f"temp_dir 是这个 {temp_dir}")
#         logging.debug(f'temp_dirs 是这个 {self.temp_dirs}')
#         self.img_dir_ls = glob.glob(os.path.join(temp_dir, "*"")
#         # self.img_dir_ls = self.temp_dirs
#         logging.debug(f'img_dirs_ls 是这个 {self.img_dir_ls}')

#         # Debug: Print index and length of img_dir_ls
#         logging.debug(f"Index: {index}, Length of img_dir_ls: {len(self.img_dir_ls)}")
#         # self.img_dir_ls 是一个list， 是一个video所有image 
#         # 但是我需要self.img_dir_ls 更上一层，是所有video的合集 
        
#         item = super(VATData, self).__getitem__(index)  # Call the __getitem__ method of VATData
#         # 这个index得是一个一个frame.jpg的index
        
#         shutil.rmtree(temp_dir)
#         return item

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# class VATTruePerData(VATTruePerData):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.zip_files = glob.glob(os.path.join(self.img_dir, "*.zip"))
#         self.temp_dirs = []
#         logging.debug(f"Detected {len(self.zip_files)} zip files in {self.img_dir}")

#     def __getitem__(self, index):
#         temp_dir = unzip_files(self.zip_files[index])
#         self.temp_dirs.append(temp_dir)
#         self.img_dir_ls = glob.glob(os.path.join(temp_dir, "*"))

#         # Debug: Print index and length of img_dir_ls
#         logging.debug(f"Index: {index}, Length of img_dir_ls: {len(self.img_dir_ls)}")

#         item = super(VATTruePerData, self).__getitem__(index)  # Call the __getitem__ method of VATTruePerData
#         shutil.rmtree(temp_dir)
#         return item

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# class FullTestVATData(FullTestVATData):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.zip_files = glob.glob(os.path.join(self.img_dir, "*.zip"))
#         self.temp_dirs = []
#         logging.debug(f"Detected {len(self.zip_files)} zip files in {self.img_dir}")

#     def __getitem__(self, index):
#         temp_dir = unzip_files(self.zip_files[index])
#         self.temp_dirs.append(temp_dir)
#         self.img_dir_ls = glob.glob(os.path.join(temp_dir, "*"))

#         # Debug: Print index and length of img_dir_ls
#         logging.debug(f"Index: {index}, Length of img_dir_ls: {len(self.img_dir_ls)}")

#         item = super(FullTestVATData, self).__getitem__(index)  # Call the __getitem__ method of FullTestVATData
#         shutil.rmtree(temp_dir)
#         return item

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# def make_data_loader(cfg, mode="train"):
#     assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = set_vat_transform_op()
#     temporal_transform = [TemporalRandomCrop(16)]
#     temporal_transform = TemporalCompose(temporal_transform)
#     video_loader = VideoLoader()

#     if mode == "train":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TRAIN_IMG_DATA,
#             cfg.TRAIN_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "valid":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.VALID_IMG_DATA,
#             cfg.VALID_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "trainval":
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TRAINVAL_IMG_DATA,
#             cfg.TRAINVAL_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     else:
#         data_set = VATData(
#             cfg.DATA_ROOT,
#             cfg.TEST_IMG_DATA,
#             cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=cfg.TRAIN_BATCH_SIZE,
#         shuffle=cfg.SHUFFLE,
#         num_workers=cfg.NUM_WORKERS,
#     )
#     return data_loader

# @DATA_LOADER_REGISTRY.register()
# def vat_data_loader(cfg, mode="train"):
#     assert (mode in ["train", "valid", "trainval", "test", "full_test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = build_transform_spatial(cfg)
#     temporal_transform = [TemporalDownsample(length=100), TemporalRandomCrop(16)]
#     temporal_transform = TemporalCompose(temporal_transform)

#     data_cfg = cfg.DATA
#     if "face" in data_cfg.TRAIN_IMG_DATA:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
#     else:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

#     if mode == "train":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TRAIN_IMG_DATA,
#             data_cfg.TRAIN_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "valid":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.VALID_IMG_DATA,
#             data_cfg.VALID_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "trainval":
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TRAINVAL_IMG_DATA,
#             data_cfg.TRAINVAL_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     elif mode == "full_test":
#         temporal_transform = [TemporalDownsample(length=100), TemporalEvenCropDownsample(16, 6)]
#         temporal_transform = TemporalCompose(temporal_transform)
#         return FullTestVATData(
#             data_cfg.ROOT,
#             data_cfg.TEST_IMG_DATA,
#             data_cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )
#     else:
#         data_set = VATData(
#             data_cfg.ROOT,
#             data_cfg.TEST_IMG_DATA,
#             data_cfg.TEST_LABEL_DATA,
#             video_loader,
#             spatial_transform,
#             temporal_transform,
#         )

#     loader_cfg = cfg.DATA_LOADER
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=loader_cfg.TRAIN_BATCH_SIZE,
#         shuffle=loader_cfg.SHUFFLE,
#         num_workers=loader_cfg.NUM_WORKERS,
#     )
#     return data_loader

# @DATA_LOADER_REGISTRY.register()
# def true_per_vat_data_loader(cfg, mode="train"):
#     assert (mode in ["train", "valid", "trainval", "test", "full_test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
#     spatial_transform = build_transform_spatial(cfg)
#     temporal_transform = [TemporalRandomCrop(16)]
#     temporal_transform = TemporalCompose(temporal_transform)

#     data_cfg = cfg.DATA
#     if data_cfg.TYPE == "face":
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
#     else:
#         video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

#     data_set = VATTruePerData(
#         data_root="datasets/chalearn2021",
#         data_split=mode,
#         task=data_cfg.SESSION,
#         data_type=data_cfg.TYPE,
#         video_loader=video_loader,
#         spa_trans=spatial_transform,
#         tem_trans=temporal_transform,
#     )

#     shuffle = True if mode == "train" else False
#     loader_cfg = cfg.DATA_LOADER
#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=loader_cfg.TRAIN_BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=loader_cfg.NUM_WORKERS,
#     )
#     return data_loader

# if __name__ == "__main__":
#     import os
#     from dpcv.config.tpn_cfg import cfg

#     os.chdir("../../")

#     data_loader = make_data_loader(cfg, mode="valid")
#     for i, item in enumerate(data_loader):
#         print(item["image"].shape, item["label"].shape)
#         if i > 5:
#             break


# ------------------------ 再改一遍 ------------------------
import torch
from torch.utils.data import DataLoader
from dpcv.data.transforms.transform import set_vat_transform_op
from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData
from dpcv.data.datasets.tpn_data import TPNData as VATData
from dpcv.data.datasets.tpn_data import TPNTruePerData as VATTruePerData
from dpcv.data.datasets.tpn_data import FullTestTPNData as FullTestVATData
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop, TemporalDownsample, TemporalEvenCropDownsample
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.datasets.common import VideoLoader
import numpy as np

import glob
import zipfile
import tempfile
import shutil
import os
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def unzip_files(zip_path):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir



# 最后其实也是return image 就可以了

class VATData(VATData):
    def __init__(self, data_root, img_dir, label_file, video_loader, spatial_transform=None, temporal_transform=None):

        super().__init__(data_root, img_dir, label_file, video_loader, spatial_transform, temporal_transform)
        # img_dir = /scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data

        self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
        # logging.debug(f"self.zip_files -- {self.zip_files}")
        logging.debug(f"Detected {len(self.zip_files)} zip files in {img_dir}")

        """        self.zip_files = 
        [
        '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data/13102021_Catherine_3_chinese2_110.zip',
        '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data/23092021_2_Pierre_2_Kundyz_137.zip',
        '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data/23092021_2_Pierre_5_Akhat_9.zip',...
        ] """
        # img_dir = "/scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data"


    def __getitem__(self, index):
        img = self.get_image_data(index)
        label = self.get_ocean_label(index)
        # logging.debug(f'Image shape: {img.shape}')
        temp_dir = unzip_files(self.zip_files[index])
        shutil.rmtree(temp_dir)
        return {"image": img, "label": torch.as_tensor(label)}

    def get_image_data(self, index):
        temp_dir = unzip_files(self.zip_files[index])
        zip_path = self.zip_files[index]
        # logging.debug(f'temp_dir - {temp_dir}')
        # logging.debug(f'zip_path - {zip_path}')
        # temp_dir = '/tmpdata/tmphg_supqc"
        # zip_path = '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/train_data/23092021_Pierre_3_Isha_10.zip'
        imgs =  self.frame_sample(zip_path, temp_dir)
        return imgs



    def frame_sample(self, zip_path, temp_dir):
        # 要不要做变换，看真名/ 将什么东西放进去做变换，看假名
        if "face" in zip_path: 
            frame_indices = self.list_face_frames(temp_dir) 
        else:
            frame_indices = self.list_frames(temp_dir) 

        if self.tem_trans is not None: #做变换signal
            frame_indices = self.tem_trans(frame_indices) # 做变换
        imgs = self._loading(temp_dir, frame_indices)  # 将 img_dir 改成了 temp_dir 因为这是真的要取file
        return imgs


    def _loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spa_trans is not None:
            clip = [self.spa_trans(img) for img in clip]
            # tranformation to each image in the clip list
        # logging.debug(f'Clip shape before stacking: {[img.shape for img in clip]}')  # Debug: print each image shape in clip

        clip = torch.stack(clip, 0)
        # clip = clip.permute(1, 0, 2, 3)
        # logging.debug(f'Clip tensor shape after stacking: {clip.shape}')  # Debug: print final clip tensor shape
        return clip


        
        # torch.stack(clip, 0): Stacks the list of images into a single tensor along a new dimension at index 0. The resulting tensor has the shape (num_frames, height, width, channels).
        # .permute(1, 0, 2, 3): Changes the order of dimensions to (channels, num_frames, height, width). This is often required for input to convolutional neural networks, which expect the channel dimension first.
        
        # If clip is a list of 10 images each of size (3, 224, 224) (where 3 is the number of color channels), torch.stack(clip, 0) will create a tensor of shape (10, 3, 224, 224).
        # After .permute(1, 0, 2, 3), the tensor shape becomes (3, 10, 224, 224).


    def list_frames(self,temp_dir):
        img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
        # This line uses glob.glob to find all .jpg files in the specified temp_dir directory. It returns a list of paths to these image files.
        img_path_ls = sorted(img_path_ls, key=lambda x: int(Path(x).stem[6:]))
        # logging.debug(f'img_path_ls -- {img_path_ls}')
        #For example, if the filenames are frame_0001.jpg, frame_0002.jpg, etc., this line sorts them in numerical order.
        frame_indices = [int(Path(path).stem[6:]) for path in img_path_ls]
        # logging.debug(f'frame_indices -- {frame_indices}')
        # It iterates over the sorted list of image paths, extracts the numeric part from each filename (again assuming it starts from the 7th character), converts it to an integer, and adds it to the list.
        # frame_indices = 1,2,3,4,5,6,7,...
        return frame_indices

    def list_face_frames(self, temp_dir):
        img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
        img_path_ls = sorted(img_path_ls, key=lambda x: int(Path(x).stem[5:]))
        frame_indices = [int(Path(path).stem[5:]) for path in img_path_ls]
        return frame_indices

    def __len__(self):
        return len(self.zip_files)
        
    
    @staticmethod
    def image_sample(temp_dir): 
        # logging.debug(f'Sampling image from directory: {temp_dir}')
        img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
        # 所有Temp_Dir里面每个frame的地址list
        num_img = len(img_path_ls)
        # len
        sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
        # sampling
        selected = random.choice(sample_frames)
        return img_path_ls[selected]

    def __len__(self):
        return len(self.zip_files)

                
class VATTruePerData(VATTruePerData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_files = glob.glob(os.path.join(self.img_dir, "*.zip"))
        self.temp_dirs = []
        logging.debug(f"Detected {len(self.zip_files)} zip files in {self.img_dir}")

    def __getitem__(self, index):
        temp_dir = unzip_files(self.zip_files[index])
        self.temp_dirs.append(temp_dir)
        self.img_dir_ls = glob.glob(os.path.join(temp_dir, "*"))

        # Debug: Print index and length of img_dir_ls
        logging.debug(f"Index: {index}, Length of img_dir_ls: {len(self.img_dir_ls)}")
        item = super(VATTruePerData, self).__getitem__(index)  # Call the __getitem__ method of VATTruePerData
        shutil.rmtree(temp_dir)
        return item

    def __len__(self):
        return len(self.zip_files)

    def __del__(self):
        if hasattr(self, 'temp_dirs'):
            for temp_dir in self.temp_dirs:
                shutil.rmtree(temp_dir)

class FullTestVATData(FullTestVATData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_files = glob.glob(os.path.join(self.img_dir, "*.zip"))
        self.temp_dirs = []
        logging.debug(f"Detected {len(self.zip_files)} zip files in {self.img_dir}")

    def __getitem__(self, index):
        temp_dir = unzip_files(self.zip_files[index])
        self.temp_dirs.append(temp_dir)
        self.img_dir_ls = glob.glob(os.path.join(temp_dir, "*"))

        # Debug: Print index and length of img_dir_ls
        logging.debug(f"Index: {index}, Length of img_dir_ls: {len(self.img_dir_ls)}")

        item = super(FullTestVATData, self).__getitem__(index)  # Call the __getitem__ method of FullTestVATData
        shutil.rmtree(temp_dir)
        return item

    def __len__(self):
        return len(self.zip_files)

    def __del__(self):
        if hasattr(self, 'temp_dirs'):
            for temp_dir in self.temp_dirs:
                shutil.rmtree(temp_dir)

def make_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = set_vat_transform_op()
    temporal_transform = [TemporalRandomCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)
    video_loader = VideoLoader()

    if mode == "train":
        data_set = VATData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = VATData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = VATData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = VATData(
            cfg.DATA_ROOT,
            cfg.TEST_IMG_DATA,
            cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
    )
    return data_loader

@DATA_LOADER_REGISTRY.register()
def vat_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalDownsample(length=100), TemporalRandomCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA
    if "face" in data_cfg.TRAIN_IMG_DATA:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    if mode == "train":
        data_set = VATData(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = VATData(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = VATData(
            data_cfg.ROOT,
            data_cfg.TRAINVAL_IMG_DATA,
            data_cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "full_test":
        temporal_transform = [TemporalDownsample(length=100), TemporalEvenCropDownsample(16, 6)]
        temporal_transform = TemporalCompose(temporal_transform)
        return FullTestVATData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = VATData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )

    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        shuffle=loader_cfg.SHUFFLE,
        num_workers=loader_cfg.NUM_WORKERS,
    )
    return data_loader

@DATA_LOADER_REGISTRY.register()
def true_per_vat_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalRandomCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA
    if data_cfg.TYPE == "face":
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    data_set = VATTruePerData(
        data_root="datasets/chalearn2021",
        data_split=mode,
        task=data_cfg.SESSION,
        data_type=data_cfg.TYPE,
        video_loader=video_loader,
        spa_trans=spatial_transform,
        tem_trans=temporal_transform,
    )

    shuffle = True if mode == "train" else False
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=loader_cfg.NUM_WORKERS,
    )
    return data_loader

if __name__ == "__main__":
    import os
    from dpcv.config.tpn_cfg import cfg

    os.chdir("../../")

    data_loader = make_data_loader(cfg, mode="valid")
    for i, item in enumerate(data_loader):
        print(item["image"].shape, item["label"].shape)
        if i > 5:
            break
 