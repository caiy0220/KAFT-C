from utils import data_process as dp
from utils import my_models
import torch, torchvision
from torch import nn

DS_MNAME_DICT = {
    'MNIST': 'CNN',
    'CIFAR10': 'WRN40_4'
}

DS_DICT = {
    'MNIST': torchvision.datasets.MNIST,
    'CIFAR10': torchvision.datasets.CIFAR10,
}

V_DEF_DICT = {
    'MNIST': -dp.MNIST_PIX_MEAN / dp.MNIST_PIX_STD,
    'CIFAR10': 0,
}

M_NAME_DICT = {
    'CNN': my_models.CNN,
    'WRN40_4': my_models.WRN40_4,
}

IMG_SIZE_DICT = {
    'CNN': (1, 28, 28),
    'WRN40_4': (3, 32, 32),    # CIFAR10
}

CLASS_NUM_DICT = {
    'MNIST': 10,
    'CIFAR10': 10,
}

TEST_TRANSFORM_DICT = {
    'MNIST': dp.get_transforms_gray,
    'CIFAR10': dp.get_transforms_rgb,
}

TRAIN_TRANSFORM_DICT = {
    'MNIST': dp.get_transforms_gray,
    'CIFAR10': dp.get_transforms_rgb_simpleAug,
}

PIX_STATS_DICT = {
    'MNIST': (dp.MNIST_PIX_MEAN, dp.MNIST_PIX_STD),
    'CIFAR10': (dp.CIFAR10_PIX_MEAN, dp.CIFAR10_PIX_STD),
}

def get_stats4transform(ds_name, m_name):
    img_size = IMG_SIZE_DICT[m_name]
    pix_stats = PIX_STATS_DICT[ds_name] # get pix mean and std
    return img_size, pix_stats

# Wrap model with softmax layer
class ModelWrapper(nn.Module):
    def __init__(self, m_name, device, pth=None, **kwargs):
        super().__init__()
        self.m = M_NAME_DICT[m_name](**kwargs)
        if pth is not None:
            self.m.load_state_dict(torch.load(pth))
        self.m.eval(), self.m.to(device)

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)
