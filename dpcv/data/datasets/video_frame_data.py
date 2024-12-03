# import random
# import torch
# from torch.utils.data import DataLoader
# from PIL import Image
# import numpy as np
# import glob
# import zipfile
# import tempfile
# import shutil
# from dpcv.data.datasets.bi_modal_data import VideoData
# from dpcv.data.transforms.transform import set_transform_op
# from dpcv.data.transforms.build import build_transform_spatial
# from .build import DATA_LOADER_REGISTRY
# import os
# import logging

# def unzip_files(zip_path):
#     temp_dir = tempfile.mkdtemp()
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(temp_dir)
#     return temp_dir

# class SingleFrameData(VideoData):
#     def __init__(self, data_root, img_dir, label_file, trans=None):
#         super().__init__(data_root, img_dir, label_file)  # Ensure correct superclass initialization
#         self.trans = trans
#         self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
#         self.temp_dirs = []
#         print(f"Detected {len(self.zip_files)} zip files in {img_dir}")
#         print("Directory listing:", os.listdir(img_dir))  # Print directory contents

#     def __getitem__(self, index):
#         img = self.get_image_data(index)
#         label = self.get_ocean_label(index)

#         if self.trans:
#             img = self.trans(img)

#         return {"image": img, "label": torch.as_tensor(label)}

#     def get_image_data(self, index):
#         zip_path = self.zip_files[index]
#         temp_dir = unzip_files(zip_path)
#         self.temp_dirs.append(temp_dir)
#         img_path = self.image_sample(temp_dir)
#         img = Image.open(img_path).convert("RGB")
#         return img

#     @staticmethod
#     def image_sample(temp_dir):
#         img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
#         num_img = len(img_path_ls)
#         sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
#         selected = random.choice(sample_frames)
#         return img_path_ls[selected]

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# class AllSampleFrameData(VideoData):
#     def __init__(self, data_root, img_dir, label_file, trans=None, length=100):
#         super().__init__(data_root, img_dir, label_file)  # Ensure correct superclass initialization
#         self.trans = trans
#         self.len = length
#         self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
#         self.temp_dirs = []
#         print(f"Detected {len(self.zip_files)} zip files in {img_dir}")
#         print("Directory listing:", os.listdir(img_dir))  # Print directory contents

#     def __getitem__(self, idx):
#         img_obj_ls = self.get_sample_frames(idx)
#         label = self.get_ocean_label(idx)
#         if self.trans is not None:
#             img_obj_ls = [self.trans(img) for img in img_obj_ls]
#         return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

#     def get_sample_frames(self, idx):
#         zip_path = self.zip_files[idx]
#         temp_dir = unzip_files(zip_path)
#         self.temp_dirs.append(temp_dir)
#         img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
#         sample_frames_id = np.linspace(0, len(img_path_ls), self.len, endpoint=False, dtype=np.int16).tolist()
#         img_path_ls_sampled = [img_path_ls[idx] for idx in sample_frames_id]
#         img_obj_ls = [Image.open(img_path) for img_path in img_path_ls_sampled]
#         return img_obj_ls

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# class AllSampleFrameData2(VideoData):
#     def __init__(self, data_root, img_dir, label_file, trans=None):
#         super().__init__(data_root, img_dir, label_file)  # Ensure correct superclass initialization
#         self.trans = trans
#         self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
#         self.temp_dirs = []
#         print(f"Detected {len(self.zip_files)} zip files in {img_dir}")
#         print("Directory listing:", os.listdir(img_dir))  # Print directory contents

#     def __getitem__(self, idx):
#         img_obj_ls = self.get_sample_frames(idx)
#         label = self.get_ocean_label(idx)
#         if self.trans is not None:
#             img_obj_ls = [self.trans(img) for img in img_obj_ls]
#         return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

#     def get_sample_frames(self, idx):
#         zip_path = self.zip_files[idx]
#         temp_dir = unzip_files(zip_path)
#         self.temp_dirs.append(temp_dir)
#         img_path_ls = sorted(glob.glob(f"{temp_dir}/*.jpg"))
#         img_obj_ls = [Image.open(img_path) for img_path in img_path_ls]
#         return img_obj_ls

#     def __len__(self):
#         return len(self.zip_files)

#     def __del__(self):
#         if hasattr(self, 'temp_dirs'):
#             for temp_dir in self.temp_dirs:
#                 shutil.rmtree(temp_dir)

# @DATA_LOADER_REGISTRY.register()
# def single_frame_data_loader(cfg, mode="train"):

#     assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
#         "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
#     shuffle = cfg.DATA_LOADER.SHUFFLE
#     transform = build_transform_spatial(cfg)
#     if mode == "train":
#         data_set = SingleFrameData(
#             cfg.DATA.ROOT,  # Ensure correct path to data root
#             cfg.DATA.TRAIN_IMG_DATA,
#             cfg.DATA.TRAIN_LABEL_DATA,
#             transform,
#         )
#     elif mode == "valid":
#         data_set = SingleFrameData(
#             cfg.DATA.ROOT,  # Ensure correct path to data root
#             cfg.DATA.VALID_IMG_DATA,
#             cfg.DATA.VALID_LABEL_DATA,
#             transform,
#         )
#         shuffle = False
#     elif mode == "trainval":
#         data_set = SingleFrameData(
#             cfg.DATA.ROOT,  # Ensure correct path to data root
#             cfg.DATA.TRAINVAL_IMG_DATA,
#             cfg.DATA.TRAINVAL_LABEL_DATA,
#             transform,
#         )
#     elif mode == "full_test":
#         return AllSampleFrameData(
#             cfg.DATA.ROOT,  # Ensure correct path to data root
#             cfg.DATA.TEST_IMG_DATA,
#             cfg.DATA.TEST_LABEL_DATA,
#             transform,
#         )
#     else:
#         data_set = SingleFrameData(
#             cfg.DATA.ROOT,  # Ensure correct path to data root
#             cfg.DATA.TEST_IMG_DATA,
#             cfg.DATA.TEST_LABEL_DATA,
#             transform,
#         )
#         shuffle = False

#     data_loader = DataLoader(
#         dataset=data_set,
#         batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#     )
#     return data_loader

import random
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import glob
import zipfile
import tempfile
import shutil
import logging
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to unzip files and return a temporary directory path
# def unzip_files(zip_path):
#     print(f'Unzipping file: {zip_path}')
#     temp_dir = tempfile.mkdtemp()  # Create a temporary directory

#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(temp_dir)
        
#     print("\n temp_dir==> \n", temp_dir)
#     print("\n inside temp_dir==> \n")

#     for root, dirs, files in os.walk(temp_dir):
#         print(f"Directory: {root}")
#         # for file in files:
#         #     print(f"File: {os.path.join(root, file)}")

#     # extracted_files = glob.glob(os.path.join(temp_dir, "**/*.jpg"), recursive=True)
#     # print("\n Found jpg files==> \n", len(extracted_files))
    
#     extracted_files = glob.glob(os.path.join(temp_dir, "*.jpg")) 这是原来的代码,work for frame data
#     # print(f"Extracted files: {extracted_files}")
#     if not extracted_files:
#         raise ValueError(f"No images found in zip file {zip_path}")
#     # logging.debug(f'Created temporary directory: {temp_dir}')
#     # return temp_dir 这是原来的代码，work for frame data

#     return temp_dir

def unzip_files(zip_path):
    """
    Unzip files and return appropriate directory path and extracted files.
    
    Args:
        zip_path (str): Path to the zip file
        
    Returns:
        tuple: (working_dir, extracted_files) where:
            - working_dir is either temp_dir or its first subdirectory containing images
            - extracted_files is a list of paths to jpg files
    """
    # print(f'Unzipping file: {zip_path}')
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory

    try:
        # Extract files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # print("\ntemp_dir==> \n", temp_dir)
        # print("\ninside temp_dir==> \n")

        # Check directory structure
        subdirs = [d for d in os.listdir(temp_dir) 
                  if os.path.isdir(os.path.join(temp_dir, d))]
        
        # Determine working directory and search pattern
        if subdirs:
            # Case 1: Files are in a subdirectory
            working_dir = os.path.join(temp_dir, subdirs[0])
            # print(f"Working from subdirectory: {working_dir}")
            extracted_files = glob.glob(os.path.join(working_dir, "*.jpg"))
        else:
            # Case 2: Files are directly in temp_dir
            working_dir = temp_dir
            # print(f"Working from root directory: {working_dir}")
            extracted_files = glob.glob(os.path.join(temp_dir, "*.jpg"))

        # Print directory structure for debugging
        for root, dirs, files in os.walk(working_dir):
            print(f"Directory: {root}")
            # Uncomment next lines for detailed file listing if needed
            # for file in files:
            #     print(f"File: {os.path.join(root, file)}")

        # print(f"\nFound {len(extracted_files)} jpg files")

        # Verify we found some files
        if not extracted_files:
            # raise ValueError(f"No jpg files found in zip file: {zip_path}")
            print(f"No jpg files found in zip file: {zip_path}")
            return None
        return working_dir

    # except Exception as e:
    #     # Clean up temp directory in case of error
    #     if os.path.exists(temp_dir):
    #         import shutil
    #         shutil.rmtree(temp_dir)
    #     raise Exception(f"Error processing zip file {zip_path}: {str(e)}")
    except Exception as e:
        print(f"Error processing zip file {zip_path}: {str(e)}")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return None, None



class SingleFrameData(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans
        self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
        logging.debug(f"Detected {len(self.zip_files)} zip files in {img_dir}")
        # logging.debug("Directory listing: {}".format(os.listdir(img_dir)))

    def __getitem__(self, index):
        # logging.debug(f'Processing index: {index}')
        temp_dir = unzip_files(self.zip_files[index])
        # logging.debug(f'Temporary directory created: {temp_dir}')
        img_path = self.image_sample(temp_dir)
        # logging.debug(f'image path -- {img_path}')
        img = Image.open(img_path).convert("RGB")
        # logging.debug(f'最后return的image -- {img}')
        label = self.get_ocean_label(index)
        if self.trans:
            img = self.trans(img)
        # logging.debug('Returning image and label')
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        # logging.debug(f'image: {img}')
        return {"image": img, "label": torch.as_tensor(label)}

    # @staticmethod
    # def image_sample(temp_dir):
    #     # logging.debug(f'Sampling image from directory: {temp_dir}')
    #     img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
    #     if not img_path_ls:
    #         raise ValueError(f"No images found in the directory {temp_dir}")
    #     num_img = len(img_path_ls)
    #     sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
    #     selected = random.choice(sample_frames)
    #     print(f"Available images: {len(img_path_ls)}, Selected index: {selected}")
    #     return img_path_ls[selected]
    @staticmethod
    def image_sample(temp_dir):
        if temp_dir is None:
            print("Skipping: Directory is None")
            return None

        img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
        
        # Return None if no images found
        if not img_path_ls:
            print(f"No images found in the directory {temp_dir}")
            return None
        
        # Process only if images are found
        num_img = len(img_path_ls)
        sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
        selected = random.choice(sample_frames)
        print(f"Available images: {len(img_path_ls)}, Selected index: {selected}")
        return img_path_ls[selected]

    def __len__(self):
        return len(self.zip_files)

class AllSampleFrameData(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None, length=100):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans
        self.len = length
        self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
        # logging.debug(f"Detected {len(self.zip_files)} zip files in {img_dir}")
        # logging.debug("Directory listing: {}".format(os.listdir(img_dir)))

    def __getitem__(self, idx):
        # logging.debug(f'Processing index: {idx}')
        temp_dir = unzip_files(self.zip_files[idx])
        # logging.debug(f'Temporary directory created: {temp_dir}')
        img_obj_ls = self.get_sample_frames(temp_dir)
        # logging.debug(f'Selected image paths: {img_obj_ls}')
        label = self.get_ocean_label(idx)
        if self.trans is not None:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        # logging.debug('Returning images and label')
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        # return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

    def get_sample_frames(self, temp_dir):
        # logging.debug(f'Sampling frames from directory: {temp_dir}')
        img_path_ls = glob.glob(f"{temp_dir}/*.jpg")
        sample_frames_id = np.linspace(0, len(img_path_ls), self.len, endpoint=False, dtype=np.int16).tolist()
        img_path_ls_sampled = [img_path_ls[idx] for idx in sample_frames_id]
        img_obj_ls = [Image.open(img_path) for img_path in img_path_ls_sampled]
        return img_obj_ls

    def __len__(self):
        return len(self.zip_files)

class AllSampleFrameData2(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans
        self.zip_files = glob.glob(os.path.join(img_dir, "*.zip"))
        logging.debug(f"Detected {len(self.zip_files)} zip files in {img_dir}")
        # logging.debug("Directory listing: {}".format(os.listdir(img_dir)))

    def __getitem__(self, idx):
        # logging.debug(f'Processing index: {idx}')
        temp_dir = unzip_files(self.zip_files[idx])
        # logging.debug(f'Temporary directory created: {temp_dir}')
        img_obj_ls = self.get_sample_frames(temp_dir)
        # logging.debug(f'Selected image paths: {img_obj_ls}')
        label = self.get_ocean_label(idx)
        if self.trans is not None:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        # logging.debug('Returning images and label')
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

    def get_sample_frames(self, temp_dir):
        # logging.debug(f'Sampling frames from directory: {temp_dir}')
        img_path_ls = sorted(glob.glob(f"{temp_dir}/*.jpg"))
        img_obj_ls = [Image.open(img_path) for img_path in img_path_ls]
        return img_obj_ls

    def __len__(self):
        return len(self.zip_files)

@DATA_LOADER_REGISTRY.register()
def single_frame_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE
    transform = build_transform_spatial(cfg)
    if mode == "train":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,  # Ensure correct path to data root
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,  # Ensure correct path to data root
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transform,
        )
        shuffle = False
    elif mode == "trainval":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,  # Ensure correct path to data root
            cfg.DATA.TRAINVAL_IMG_DATA,
            cfg.DATA.TRAINVAL_LABEL_DATA,
            transform,
        )
    elif mode == "full_test":
        return AllSampleFrameData(
            cfg.DATA.ROOT,  # Ensure correct path to data root
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
    else:
        data_set = SingleFrameData(
            cfg.DATA.ROOT,  # Ensure correct path to data root
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
        shuffle = False

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader
