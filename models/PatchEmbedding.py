import h5py
import numpy as np
from tqdm import tqdm

class PatchEmbedding():
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.Origin_Data_Load()

    def Origin_Data_Load(self):
        # Tqdm으로 데이터 로드
        with h5py.File(self.args.input, "r") as f:
            # 데이터셋 크기 확인
            num_samples = len(f["images"])  # 이미지 개수
            
            # 빈 배열 생성 (데이터 저장용)
            rgb_origin = np.zeros(f["images"].shape, dtype=f["images"].dtype)
            depth_origin = np.zeros(f["depths"].shape, dtype=f["depths"].dtype)
            label_origin = np.zeros(f["labels"].shape, dtype=f["labels"].dtype)

            # tqdm을 사용하여 데이터 로드 진행 표시
            for i in tqdm(range(num_samples), desc="Loading HDF5 Data"):
                rgb_origin[i] = f["images"][i]
                depth_origin[i] = f["depths"][i]
                label_origin[i] = f["labels"][i]

        print(f"="*50)
        print("RGB shape:", rgb_origin.shape)
        print("Depth shape:", depth_origin.shape)
        print("Label shape:", label_origin.shape)
        print(f"="*50)
        print(f"RGB dtype: {rgb_origin.dtype}")
        print(f"Depth dtype: {depth_origin.dtype}")
        print(f"Label dtype: {label_origin.dtype}")
        print(f"="*50)
        print(f"num-classes: {len(np.unique(label_origin))}")
        print(f"="*50)

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class RGBDepthFromListDataset(Dataset):
    def __init__(self, args):
        self.base_folder = args.input_dir
        self.txt_path = args.txt_path
        self.pairs = []

        with open(self.txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                depth_rel, rgb_rel, _ = line.split()
                self.pairs.append((depth_rel, rgb_rel))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        depth_rel_path, rgb_rel_path = self.pairs[idx]

        depth_path = os.path.join(self.base_folder, depth_rel_path)
        rgb_path = os.path.join(self.base_folder, rgb_rel_path)

        depth = Image.open(depth_path)
        rgb = Image.open(rgb_path).convert('RGB')

        depth_tensor = transforms.ToTensor()(depth)
        rgb_tensor = transforms.ToTensor()(rgb)

        combined = torch.cat([rgb_tensor, depth_tensor], dim=0)
        return combined