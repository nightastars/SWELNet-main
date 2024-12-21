import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_augment import augment_img


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, augment, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_source.npy')))   # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path = sorted(glob(os.path.join(saved_path, '*_label.npy')))
        unlabel_path = sorted(glob(os.path.join(saved_path, '*_unlabel.npy')))
        # input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        # target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        # unlabel_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        self.augment = augment

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            unlabel_ = [f for f in unlabel_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
                self.unlabel_ = unlabel_
            else:  # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
                self.unlabel_ = [np.load(f) for f in unlabel_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            unlabel_ = [f for f in unlabel_path if test_patient in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
                self.unlabel_ = unlabel_
            else:    # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
                self.unlabel_ = [np.load(f) for f in unlabel_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img, unlabel_img = self.input_[idx], self.target_[idx], self.unlabel_[idx]
        if self.load_mode == 0:
            input_img, target_img, unlabel_img = np.load(input_img), np.load(target_img), np.load(unlabel_img)

        if self.augment:
            temp = np.random.randint(0, 8)
            input_img, target_img, unlabel_img = augment_img(input_img, temp), augment_img(target_img, temp), augment_img(unlabel_img, temp)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            unlabel_img = self.transform(unlabel_img)

        if self.patch_size:
            input_patches, target_patches, unlabel_patches = get_patch(input_img,
                                                      target_img,
                                                      unlabel_img,
                                                      self.patch_n,
                                                      self.patch_size)

            return (input_patches, target_patches, unlabel_patches)
        else:
            return (input_img, target_img, unlabel_img)


def get_patch(full_input_img, full_target_img, full_unlabel_img, patch_n, patch_size):  # 定义patch
    assert full_input_img.shape == full_target_img.shape and full_target_img.shape == full_unlabel_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    patch_unlabel_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_unlabel_img = full_unlabel_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        patch_unlabel_imgs.append(patch_unlabel_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs), np.array(patch_unlabel_imgs)


def get_loader(mode='train', load_mode=0, augment=True,
               saved_path=None, test_patient='TEST',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=32):
    dataset_ = ct_dataset(mode, load_mode, augment, saved_path, test_patient, patch_n, patch_size, transform)
    # shuffle将序列的所有元素随机排序
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True if mode=='train' else False, num_workers=num_workers, pin_memory=True)   # shuffle=True
    return data_loader