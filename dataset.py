import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch, is_jpg_file
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
# ----------------------------------------------------------------
# 文件名“匹配键”——兼容 SRD 等数据集中 GT(C) 命名带后缀的情况
# （如 train_C: xxx_no_shadow.jpg, test_C: xxx_free.jpg 等）
_GT_SUFFIXES = ('_no_shadow', '_free', '_shadow_free', '_nostadow',
                '_shadowfree', '_gt')

def _match_key(filename):
    """去掉扩展名，并剥离已知 GT 后缀，得到可与 A/B 对齐的匹配键。"""
    base = os.path.splitext(filename)[0]
    low = base.lower()
    for s in _GT_SUFFIXES:
        if low.endswith(s):
            return base[: len(base) - len(s)]
    return base

def _aligned_filenames(rgb_dir, gt_dir, input_dir, mask_dir):
    """按匹配键对齐 A/B/C 三组文件名，返回 (clean, noisy, mask) 路径列表。"""
    def _list(d):
        return [x for x in sorted(os.listdir(os.path.join(rgb_dir, d)))
                if (is_png_file(x) or is_jpg_file(x))]
    clean_all, noisy_all, mask_all = _list(gt_dir), _list(input_dir), _list(mask_dir)
    clean_map = {_match_key(x): x for x in clean_all}
    noisy_map = {_match_key(x): x for x in noisy_all}
    mask_map  = {_match_key(x): x for x in mask_all}
    keys = sorted(clean_map.keys() & noisy_map.keys() & mask_map.keys())
    if not keys:  # 退化：退回按索引对齐（保持原有行为，兼容 ISTD 等严格同名数据）
        n = min(len(clean_all), len(noisy_all), len(mask_all))
        keys = list(range(n))
        clean_map = {i: clean_all[i] for i in range(n)}
        noisy_map = {i: noisy_all[i] for i in range(n)}
        mask_map  = {i: mask_all[i]  for i in range(n)}
    clean = [os.path.join(rgb_dir, gt_dir,    clean_map[k]) for k in keys]
    noisy = [os.path.join(rgb_dir, input_dir, noisy_map[k]) for k in keys]
    mask  = [os.path.join(rgb_dir, mask_dir,  mask_map[k])  for k in keys]
    assert len(clean) == len(noisy) == len(mask) and len(keys) > 0, \
        "A/B/C 三组文件无法按文件名对齐，请检查数据集结构"
    return clean, noisy, mask

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'

        self.clean_filenames, self.noisy_filenames, self.mask_filenames = \
            _aligned_filenames(rgb_dir, gt_dir, input_dir, mask_dir)
        

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], aug=True)
        mask = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        mask = mask[r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        
        ############## data aug ############
        p = 0.3
        hue_interval = [-p, p]
        saturation_interval = [1-p, 1+p]
        fn_ids = torch.randperm(2)
        h_value = torch.empty(1).uniform_(hue_interval[0], hue_interval[1])
        s_value = torch.empty(1).uniform_(saturation_interval[0], saturation_interval[1])
        for fn_id in fn_ids:
            if fn_id == 0:
                clean = TF.adjust_saturation(clean, s_value)
                noisy = TF.adjust_saturation(noisy, s_value)
            if fn_id == 1:
                clean = TF.adjust_hue(clean, h_value)
                noisy = TF.adjust_hue(noisy, h_value)
        
        
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'

        self.clean_filenames, self.noisy_filenames, self.mask_filenames = \
            _aligned_filenames(rgb_dir, gt_dir, input_dir, mask_dir)
        


        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index], aug=False)
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename, mask_filename

if __name__ == '__main__':
    print(transforms_aug)