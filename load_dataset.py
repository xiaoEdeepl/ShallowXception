import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os


img_size = 299  # 图像大小
transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])

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
        tensor_label = torch.tensor(label)
        return img, tensor_label


class ff_dataset(Dataset):
    def __init__(self, data_dir, label_dir, transform, label_dict, classes=5):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_dict = label_dict
        self.file_list = []
        self.classes = classes
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

        if(self.classes == 2):
            if self.label_dir in ('df', 'f2f', 'fshift', 'fswap', 'nt', 'fake'):
                label = 0
            elif self.label_dir == 'real':
                label = 1
        elif(self.classes == 5):
            label = self.label_dict[self.label_dir]

        img_tensor = self.transform(img)
        label_tensor = torch.tensor(label)
        return img_tensor, label_tensor

def ff_data_load(data_dir, transform, classes=5):
    label_dict = {'df':0, 'f2f':1, 'fshift':2, 'fswap':3, 'nt':4, 'real':5}
    fake_path = os.path.join(data_dir, 'fake')
    df = ff_dataset(fake_path, 'df', transform, label_dict, classes)
    f2f = ff_dataset(fake_path, 'f2f', transform, label_dict, classes)
    fshift = ff_dataset(fake_path, 'fshift', transform, label_dict, classes)
    fswap = ff_dataset(fake_path, 'fswap', transform, label_dict, classes)
    nt = ff_dataset(fake_path, 'nt', transform, label_dict, classes)
    fake = df + f2f + fshift + fswap + nt
    print("标签为[fake]的数据有{}".format(fake.__len__()))

    real = ff_dataset(data_dir, 'real', transform, label_dict, classes)
    print("标签为[real]的数据有{}".format(real.__len__()))
    data = fake + real

    return data


def dfdc_data_load(dataset_path, transformer):

    dataset = dfdc_dataset(dataset_path, './metadata.csv', transform=transformer)
    print(f'test set length:{len(dataset)}')

    return dataset

def test_data_load(bs):
    dataset_path = './dataset/FF++'
    batch_size = bs

    dataset = ff_data_load(dataset_path, transformer, 2)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return test_loader

def train_data_load(bs):
    # 设置路径
    # dataset_path = './dataset/FF++/'  # 根文件夹路径
    dataset_path = './dataset/dfdc'
    batch_size = bs  # 批量大小

    dataset = dfdc_data_load(dataset_path, transformer)

    total_size = len(dataset)

    num_train_set = round(0.85 * total_size)
    num_valid_set = total_size - num_train_set

    train_dataset, valide_dataset = random_split(dataset, [num_train_set, num_valid_set])

    print(f'train set length:{len(train_dataset)}, valid set length:{len(valide_dataset)}')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valide_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return train_loader, valid_loader


