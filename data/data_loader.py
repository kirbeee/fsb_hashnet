import sys, os
import re
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
from torchvision import datasets, transforms
import random
import cv2
from torch.nn import functional as F
import torch.utils.data as data
from PIL import Image

from torch.utils.data.sampler import BatchSampler
from configs.params import batch_sub, batch_samp, batch_size, seed, device, random_batch_size, test_batch_size

#### Data loader for network ####
device = device
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print('Running on device: {}'.format(device))

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class BalancedBatchSampler(BatchSampler):
    
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = torch.tensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # print(self.label_to_indices)
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):        
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # seed is used to enable same batches for both streams
            np.random.seed(self.count)
            random.seed(self.count)
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            # classes = np.random.choice(self.labels_set, self.n_classes, replace = True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def ConvertRGB2BGR(x):
    x = np.float32(x)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR) 
    return x    

def FixedImageStandard(x):
    x = (x - 127.5) * 0.0078125
    return x

class MyDataset(data.Dataset):
    def __init__(self, d_set):
        self.dataset = d_set
        
    def __getitem__(self, index):
        data, target = self.dataset[index]        
        return data, target, index

    def __len__(self):
        return len(self.dataset)

PLUSVEIN_FILENAME_RE = re.compile(
    r'^(?P<subject>\d+)_(?P<finger>[A-Za-z]+)_(?P<session>\d+)_(?P<image>\d+)\.(bmp|png|jpg|jpeg)$',
    re.IGNORECASE
)
DEFAULT_FINGER_ORDER = ['LI', 'LM', 'LR', 'RI', 'RM', 'RR']

def _finger_sort_key(code):
    code = code.upper()
    if code in DEFAULT_FINGER_ORDER:
        return (0, DEFAULT_FINGER_ORDER.index(code))
    return (1, code)

def build_plusvein_transforms(input_size, roi_size=None, augment=False):
    transform_steps = []
    if roi_size is not None:
        transform_steps.append(transforms.CenterCrop(roi_size))
    if augment:
        transform_steps.extend([
            transforms.RandomAffine(degrees=8, translate=None, scale=(0.95, 1.05), shear=0),
            transforms.RandomHorizontalFlip(p=0.3),
        ])
    transform_steps.extend([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(transform_steps)

class PlusVeinFV3Dataset(data.Dataset):
    def __init__(self, root_dir, sessions=None, class_mode='subject_finger',
                 transform=None, input_size=(112, 112), roi_size=None):
        self.root_dir = root_dir
        self.sessions = sessions
        self.class_mode = class_mode
        self.input_size = input_size
        self.roi_size = roi_size
        self.samples = []
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                if not file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    continue
                match = PLUSVEIN_FILENAME_RE.match(file_name)
                if not match:
                    continue
                subject_id = int(match.group('subject'))
                finger_code = match.group('finger').upper()
                session_id = int(match.group('session'))
                if sessions is not None and session_id not in sessions:
                    continue
                self.samples.append({
                    'path': os.path.join(root, file_name),
                    'subject': subject_id,
                    'finger': finger_code,
                    'session': session_id
                })

        self.subject_ids = sorted({sample['subject'] for sample in self.samples})
        self.subject_to_idx = {subject: idx for idx, subject in enumerate(self.subject_ids)}
        self.finger_codes = sorted({sample['finger'] for sample in self.samples}, key=_finger_sort_key)
        self.finger_to_idx = {finger: idx for idx, finger in enumerate(self.finger_codes)}

        self.paths = []
        self.labels = []
        for sample in self.samples:
            subject_idx = self.subject_to_idx[sample['subject']]
            finger_idx = self.finger_to_idx[sample['finger']]
            if class_mode == 'subject':
                label = subject_idx
            elif class_mode == 'subject_finger':
                label = subject_idx * len(self.finger_codes) + finger_idx
            else:
                raise ValueError(f'Unsupported class_mode: {class_mode}')
            self.paths.append(sample['path'])
            self.labels.append(label)

        self.num_classes = len(self.subject_ids) if class_mode == 'subject' else len(self.subject_ids) * len(self.finger_codes)
        self.transform = transform or build_plusvein_transforms(self.input_size, self.roi_size, augment=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('L')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def gen_data(path_dir, mode, type='periocular', aug='False', indexing = False):
    if mode == 'test' and aug == 'True':
        raise('Testing dataset has augmentation!')
    if type == 'face':
        sz = (112, 112)
    elif type == 'periocular' or type == 'peri':
        sz = (112, 112)
    
    data_trans = transforms.Compose( [ transforms.Resize(sz),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                    ] )
    aug_trans = transforms.Compose( [ transforms.RandomAffine(degrees=10, translate=None, scale=(1.0,1.2),
                                                                    shear=0),
                                    transforms.Resize(sz), 
                                    transforms.RandomHorizontalFlip(p=0.8), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                    ] )
        
    data_set = datasets.ImageFolder(path_dir, transform = data_trans)        
    data_sampler = BalancedBatchSampler(data_set.targets, n_classes = batch_sub, n_samples = batch_samp)
    if aug == 'True':
        data_set_aug = datasets.ImageFolder(path_dir, transform = aug_trans)
        data_sets = data_set_aug
        # data_sets = ConcatDataset(data_set, data_set_aug)
    else:
        data_sets = data_set
    
    if indexing == True:
        data_sets = MyDataset(data_sets)

    if mode == 'train':
        data_loader = torch.utils.data.DataLoader(data_sets, batch_sampler = data_sampler, num_workers = 4,
                                              worker_init_fn = random.seed(seed))
    elif mode == 'train_rand':
        data_loader = torch.utils.data.DataLoader(data_sets, batch_size = random_batch_size, num_workers = 4,
                                              worker_init_fn = random.seed(seed), shuffle = True, drop_last = True)
    elif mode == 'test' and aug == 'False':
        data_loader = torch.utils.data.DataLoader(data_sets, batch_size = test_batch_size*4, shuffle = False, 
                                                num_workers = 6, worker_init_fn = random.seed(seed))
    
    return data_loader, data_set

def gen_plusvein_data(path_dir, mode, sessions=None, class_mode='subject_finger',
                      aug='False', indexing=False, input_size=(112, 112), roi_size=None,
                      balanced=True):
    if mode == 'test' and aug == 'True':
        raise('Testing dataset has augmentation!')

    data_trans = build_plusvein_transforms(input_size, roi_size=roi_size, augment=False)
    aug_trans = build_plusvein_transforms(input_size, roi_size=roi_size, augment=True)
    transform = aug_trans if aug == 'True' else data_trans

    data_set = PlusVeinFV3Dataset(path_dir, sessions=sessions, class_mode=class_mode,
                                  transform=transform, input_size=input_size, roi_size=roi_size)

    if indexing == True:
        data_set = MyDataset(data_set)

    if mode == 'train' and balanced:
        data_sampler = BalancedBatchSampler(data_set.labels, n_classes=batch_sub, n_samples=batch_samp)
        data_loader = torch.utils.data.DataLoader(data_set, batch_sampler=data_sampler, num_workers=4,
                                                  worker_init_fn=random.seed(seed))
    else:
        shuffle = mode == 'train'
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=test_batch_size if mode == 'test' else batch_size,
                                                  shuffle=shuffle, drop_last=shuffle, num_workers=4,
                                                  worker_init_fn=random.seed(seed))

    return data_loader, data_set
