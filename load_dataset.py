import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import torch.nn.functional as F


img_size = 299  # 图像大小
transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])

class celebdf_dataset(Dataset):
    def __init__(self,  root_dir, transform=None):
        self.root_dir = root_dir
        self.filelist = []
        self.transform = transform

        label_dir = os.listdir(root_dir)
        real_name_path = root_dir + '/' + label_dir[0]
        synthesis_name_path = root_dir + '/' + label_dir[1]

        real_videos = os.listdir(real_name_path)
        synthesis_videos = os.listdir(synthesis_name_path)

        for real_video in real_videos:
            video_name_path = real_name_path + '/' + real_video
            frames = os.listdir(video_name_path)
            for frame in frames:
                frame_path = video_name_path + '/' + frame
                self.filelist.append(frame_path)

        for synthesis_video in synthesis_videos:
            video_name_path = synthesis_name_path + '/' + synthesis_video
            frames = os.listdir(video_name_path)
            for frame in frames:
                frame_path = video_name_path + '/' + frame
                self.filelist.append(frame_path)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        frame_path = self.filelist[idx]
        frame = Image.open(frame_path)
        label_temp = frame_path.split('/')[-3]
        # print(label_temp)
        if label_temp == 'real':
            label = 1
        else:
            label = 0
        tensor_frame = self.transform(frame)
        tensor_label = torch.tensor(label, dtype=torch.long)

        # tensor_label_onehot = F.one_hot(tensor_label, num_classes=2).float()

        return tensor_frame, tensor_label

class dfdc_dataset(Dataset):
    def __init__(self, data_dir, metadata_dir, transform):
        super(dfdc_dataset, self).__init__()
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.transform = transform

        self.metadata = pd.read_csv(metadata_dir)

        video_namelist = os.listdir(self.data_dir)
        # video_namelist = [item + '.mp4' for item in video_namelist]

        self.itemlist = []

        for video_name in video_namelist:
            self.frame_path = os.path.join(self.data_dir, video_name)
            frame_list = os.listdir(self.frame_path)
            for frame in frame_list:
                self.itemlist.append(os.path.join(self.frame_path, frame))

    def __len__(self):
        return len(self.itemlist)

    def __getitem__(self, idx):
        img_path = self.itemlist[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        video_name = img_path.split('\\')[-2] + '.mp4'
        strlabel = self.metadata.loc[self.metadata['filename'] == video_name, 'label'].values
        if strlabel == 'FAKE':
            label = 0
        else:
            label = 1
        tensor_label = torch.tensor(label, dtype=torch.long)
        return img, tensor_label

class ff_dataset(Dataset):
    def __init__(self, data_dir, label_dir, transform, label_dict):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_dict = label_dict
        self.file_list = []
        if self.label_dir in ('df', 'f2f', 'fshift', 'fswap', 'nt', 'fake'):
            self.method_path = os.path.join(self.data_dir, self.label_dir)
            video_name_list = os.listdir(self.method_path)
            for video_name in video_name_list:
                frame_path = os.path.join(self.method_path, video_name)
                frame_list = os.listdir(frame_path)
                for frame in frame_list:
                    self.file_list.append(os.path.join(frame_path, frame))

        elif self.label_dir == 'real':
            self.video_name_list = os.listdir(os.path.join(self.data_dir, self.label_dir))
            for video_name in self.video_name_list:
                frame_path = os.path.join(self.data_dir, self.label_dir, video_name)
                frame_list = os.listdir(frame_path)
                for frame in frame_list:
                    self.file_list.append(os.path.join(frame_path, frame))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        if self.label_dir in ('df', 'f2f', 'fshift', 'fswap', 'nt', 'fake'):
            label = 0
        elif self.label_dir == 'real':
            label = 1

        # 将图片转为tensor
        img_tensor = self.transform(img)

        # 将标签转为tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        # label_tensor = F.one_hot(label_tensor, self.classes).float()

        return img_tensor, label_tensor

def ff_data_load(data_dir, transform):
    label_dict = {'df':0, 'f2f':1, 'fshift':2, 'fswap':3, 'nt':4, 'real':5}
    fake_path = os.path.join(data_dir, 'fake')
    df = ff_dataset(fake_path, 'df', transform, label_dict)
    f2f = ff_dataset(fake_path, 'f2f', transform, label_dict)
    fshift = ff_dataset(fake_path, 'fshift', transform, label_dict)
    fswap = ff_dataset(fake_path, 'fswap', transform, label_dict)
    nt = ff_dataset(fake_path, 'nt', transform, label_dict)
    fake = df + f2f + fshift + fswap + nt
    print("标签为[fake]的数据有{}".format(fake.__len__()))

    real = ff_dataset(data_dir, 'real', transform, label_dict)
    print("标签为[real]的数据有{}".format(real.__len__()))
    data = fake + real

    return data

def dfdc_data_load(dataset_path, transformer):

    dataset = dfdc_dataset(dataset_path, './metadata.csv', transform=transformer)
    print(f'test set length:{len(dataset)}')

    return dataset

def Test_data_load(bs, classes=2):
    dataset_path = './dataset/FF++'
    batch_size = bs

    dataset = ff_data_load(dataset_path, transformer)
    total = len(dataset)
    test_data_len = round(total * 0.1)
    drop_len = total - test_data_len
    test_data,_ = random_split(dataset, [test_data_len, drop_len])
    print(f'测试集长度:{len(test_data)}')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return test_loader

def train_data_load(bs, set):
    # 设置路径
    if set == "ff":
        dataset_path = './dataset/FF++/'
        dataset = ff_data_load(dataset_path, transformer)
    elif set == "dfdc":
        dataset_path = './dataset/dfdc'
        dataset = dfdc_data_load(dataset_path, transformer)
    elif set == "cdf":
        dataset_path = './dataset/celebdf/'
        dataset = celebdf_dataset(dataset_path, transformer)
    batch_size = bs  # 批量大小

    total_size = len(dataset)

    num_train_set = round(0.85 * total_size)
    num_valid_set = total_size - num_train_set

    train_dataset, valide_dataset = random_split(dataset, [num_train_set, num_valid_set])

    print(f'train set length:{len(train_dataset)}, valid set length:{len(valide_dataset)}')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valide_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return train_loader, valid_loader


if __name__ == "__main__":
    celebdf = celebdf_dataset("./dataset/celebdf", transform=transformer)
    img = celebdf[0]
    print(img)
