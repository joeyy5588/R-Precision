import numpy as np
import logging
from torch.utils.data import DataLoader as DD
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms



class BaseDataLoader(DD):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)
        len_valid = int(self.n_samples * split)
        
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DD(sampler=self.valid_sampler, **self.init_kwargs)
        
class DataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers = 0):
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)
        
class MyDataset(Dataset):
    def __init__(self, data_dir, word2idx):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        caption_fn = 'captions_' + data_dir.split('/')[1] + '.json'
        with open(caption_fn, 'r') as file:
            self.anns = json.load(file)
        self.anns = self.anns['annotations']
        self.caption_dict = {}
        for ann in self.anns:
            if ann['image_id'] not in self.caption_dict:
                self.caption_dict[ann['image_id']] = [ann['caption']]
            else:
                self.caption_dict[ann['image_id']].append(ann['caption'])
        self.img_list = os.listdir(data_dir)
        self.word2idx = word2idx
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __len__(self):
        return len(self.img_list)

    def pad(self, arr):
        arr = np.array(arr)
        pad_length = 15
        if len(arr) < pad_length:
            arr = np.pad(arr, ((0, pad_length - len(arr))), 'constant', constant_values = 0)
        else:
            arr = arr[:pad_length]
        return arr

    def __getitem__(self, index):
        fn = self.img_list[index]
        img_fn = self.data_dir + '/' + fn
        img = np.asarray(Image.open(img_fn).resize((64, 64),Image.ANTIALIAS), dtype=np.float32)
        if len(img.shape) < 3:
            img = np.stack((img,img,img), axis=2)
        sent_idx = random.randint(0, 4)
        caption = self.caption_dict[int(fn.split('.')[0])][sent_idx]
        if caption[-1] == '.':
            caption = caption[:-1]
        caption = caption.split(' ')
        caption = [self.word2idx[x] for x in caption]
        cap_len = len(caption)
        if cap_len > 15:
            cap_len = 15
        caption = self.pad(caption)
        if self.transform:
            img = self.transform(img)

        return img, caption, cap_len

def build_loader(data_dir, w2i):
    dataset = MyDataset(data_dir, w2i)
    batch_size = 64
    shuffle = True
    num_workers = 8
    validation_split = 0
    return DataLoader(dataset, batch_size, shuffle, validation_split, num_workers)