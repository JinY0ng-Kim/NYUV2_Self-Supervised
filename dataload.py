from glob import glob
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import h5py
from scipy.io import loadmat
import torch




# 🔹 Dataset 클래스 정의 (데이터 증강 추가)
# class NYUDepthV2Dataset_Aug(Dataset):
#     def __init__(self, rgb, depth, labels, mode='train'):
#         self.rgb = rgb
#         self.depth = depth
#         self.labels = labels
#         self.mode = mode

#         # train
#         if mode=='train':
#             self.My_Transform = T.Compose([
#                 T.NumpyToPIL(),
#                 T.RandomHorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
#                 T.RandomVerticalFlip(p=0.5),
#                 T.RandomRotation(degrees=15),   # -15도 ~ 15도 회전
#                 # T.RandomResizedCrop(size=(640, 480), scale=(0.8, 1.0)),  # 80~100% 크기에서 랜덤 크롭
#                 T.ToTensor()
#             ])

#         # val
#         elif mode=='val':
#             self.My_Transform = T.Compose([
#                 T.NumpyToPIL(),
#                 T.ToTensor()
#             ])

#     def __len__(self):
#         return self.rgb.shape[0]

#     def __getitem__(self, idx):
#         # RGB, Depth, Label 가져오기
#         rgb = self.rgb[idx]
#         depth = self.depth[idx]
#         label = self.labels[idx]

#         # # Normalize
#         # rgb = rgb.astype(np.float32) / 255.0
#         depth = depth.astype(np.float32) / np.max(depth)  # 정규화

#         rgb = np.transpose(rgb, (1, 2, 0))

#         # 변환 적용
#         rgb, depth, label = self.My_Transform(rgb, depth, label)

#         label = torch.tensor(np.array(label), dtype=torch.long)  # int64로 변환
#         """
#         Augmentation을 적용하기 위해서는 label의 dtype을 변환해야 하지만,
#         CrossEntropyLoss는 int64(long)형식을 기대함
#         label은 Augmentation 어떻게 하는게 맞을까?
#         label을 float32로 변환해서 Augmentation 적용하고 다시 int64(long)형식으로 바꾸는 게 문제가 없을까?
#         """
#         if len(label.shape) == 3:  # (C, H, W) 형태라면 squeeze()
#             label = label.squeeze(0)  # (H, W)

#         return rgb, depth, label
    
class NYUDepthV2Dataset(Dataset):
    def __init__(self, rgb, depth, labels):
        self.rgb = rgb
        self.depth = depth
        self.labels = labels

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        # RGB, Depth, Label 가져오기
        rgb = self.rgb[idx]
        depth = self.depth[idx]
        label = self.labels[idx]

        # # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / np.max(depth)  # 정규화

        # rgb = np.transpose(rgb, (1, 2, 0))

        # label = torch.tensor(np.array(label), dtype=torch.long)  # int64로 변환

        rgb = torch.from_numpy(np.array(rgb))
        depth = torch.from_numpy(np.array(depth))
        label = torch.from_numpy(np.array(label).astype(np.int64))


        if len(depth.shape) == 2:  # (C, H, W) 형태라면 squeeze()
            depth = depth.unsqueeze(0)  # (H, W)

        rgbd = torch.cat([rgb, depth], dim=0)

        return rgbd, label


def Origin_Data_Load(DATA_PATH):
    # Tqdm으로 데이터 로드
    with h5py.File(DATA_PATH, "r") as f:
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



    return rgb_origin, depth_origin, label_origin

def rgb_and_depth(rgb_origin, depth_origin):

    with open("utils/train_name.txt", "r") as f:
        train_name = [int(line.strip()) for line in f if line.strip()]

    with open("utils/test_name.txt", "r") as f:
        test_name = [int(line.strip()) for line in f if line.strip()]


    rgb_train = np.array([rgb_origin[idx-1] for idx in train_name])
    rgb_val = np.array([rgb_origin[idx-1] for idx in test_name])
    depth_train = np.array([depth_origin[idx-1] for idx in train_name])
    depth_val = np.array([depth_origin[idx-1] for idx in test_name])

    return rgb_train, rgb_val, depth_train, depth_val, train_name, test_name

def get_mask(train_name, test_name, DATA_PATH):
    f = loadmat(DATA_PATH)
    f["labels40"] = np.transpose(f["labels40"], (2, 1, 0))
    # 데이터셋 크기 확인
    num_samples = len(f["labels40"])  # 이미지 개수
    
    # 빈 배열 생성 (데이터 저장용)
    label_origin = np.zeros(f["labels40"].shape, dtype=f["labels40"].dtype)

    # tqdm을 사용하여 데이터 로드 진행 표시
    for i in tqdm(range(num_samples), desc="Loading HDF5 Data"):
        label_origin[i] = f["labels40"][i]
    
    masks_train = np.array([label_origin[idx-1] for idx in train_name])
    masks_val = np.array([label_origin[idx-1] for idx in test_name])
    return label_origin, masks_train, masks_val
        

def Load_data(args):
    DIR_PATH = args.input_dir
    ORIGIN_PATH = DIR_PATH + '/nyu_depth_v2_labeled.mat'
    MASK_PATH = DIR_PATH + "/labels40.mat"

    rgb_origin, depth_origin, label_origin = Origin_Data_Load(ORIGIN_PATH)
    rgb_train, rgb_val, depth_train, depth_val, train_name, test_name = rgb_and_depth(rgb_origin, depth_origin)
    label_40, masks_train, masks_val = get_mask(train_name, test_name, MASK_PATH)


    print(f"="*50)
    print("RGB shape:", rgb_origin.shape)
    print("Depth shape:", depth_origin.shape)
    print("Label shape:", label_40.shape)
    print(f"="*50)
    print(f"RGB dtype: {rgb_origin.dtype}")
    print(f"Depth dtype: {depth_origin.dtype}")
    print(f"Label dtype: {label_40.dtype}")
    print(f"="*50)
    print(f"num-classes: {len(np.unique(label_40))}")
    print(f"="*50)

    return rgb_train, rgb_val, depth_train, depth_val, masks_train, masks_val
