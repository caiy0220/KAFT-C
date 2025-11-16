import torch, torchvision, pickle
from torchvision import transforms
from typing import Tuple, Any
from utils import my_models
from utils.basic import log
from torch import nn
import os

MNIST_IMG_SIZE = (1, 28, 28)
MNIST_PIX_MEAN, MNIST_PIX_STD = 0.1307, 0.3081

CIFAR10_IMG_SIZE = (3, 32, 32)
CIFAR10_PIX_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR10_PIX_STD = torch.tensor((0.2023, 0.1994, 0.2010))

M_NAME_DICT = {
    'CNN': my_models.CNN,
    'WRN40_4': my_models.WRN40_4,
}


# preprocessing
def get_transforms_gray(img_size, pix_mean, pix_std):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(pix_mean, pix_std),
    ])

def get_transforms_rgb(img_size, pix_mean, pix_std):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(pix_mean, pix_std),
    ])

def get_transforms_rgb_simpleAug(img_size, pix_mean, pix_std, padding=4):
    return transforms.Compose([
        transforms.RandomCrop(img_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(pix_mean, pix_std),
    ])

def channels_last_collate(batch):
    images, labels = zip(*batch)  # Unpack batch
    images = torch.stack(images).to(memory_format=torch.channels_last)  
    labels = torch.tensor(labels)  
    return images, labels

# Denormalize for visualization
def denormalize(img, mean, std):
    # NOTE: assuming img in shape [C, H, W]
    return img.permute(1,2,0) * std + mean

""" =============> Used for explaining <============= """
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


""" 
    =============> Used for Retraining <============= 
    A customized class of ImageFolder for efficient input manipulation
"""
class ImageFolderManipulate(torchvision.datasets.ImageFolder):
    def __init__(self, root, rank_dir, rank_type, num_remove, 
                 img_size, v_default=0, flag_keep=True, 
                 **kwargs,):
        """
        Parameters:
        -----
        root:       [str]   Folder of data storage as in ImageFolder
        rank_dir:   [str]   folder of explanations saved in form of ranking
        rank_type:  [str]   Postfix of feature ranking file
                            taking value from {'_rank', '_rank_abs'}
        num_remove: [int]   Number of features to mask out
        img_size:   [tuple] Size of input image, in the order of (C,H,W)
        v_default:  [int]   Default value for replacement 
        flag_keep:  [bool]  Order of feature removal
                            False: highest-first removal
                            True:  lowest-first removal
        kwargs:     standard ImageFolder arguments, e.g. transform
        """
        super().__init__(root, **kwargs)
        self.rank_dir, self.rank_type = rank_dir, rank_type
        self.num_remove, self.flag_keep = num_remove, flag_keep
        self.v_default, self.img_size = v_default, img_size
        self._filter_invalid_entries()

    """ 
    Filtering out dataset entries without an explanation
    in case only a subset was explained.
    """
    def _filter_invalid_entries(self):
        samples = []
        for p in self.samples:
            path, _ = p
            id = path.split("/")[-1].split(".")[0]
            expl_pth = f'{self.rank_dir}/{id}.pkl'
            if os.path.exists(expl_pth):
                samples.append(p)
        self.samples = samples
        log(f'#Valid entries in {self.root}: [{len(self.samples)}]')

    """ Overwrite __getitem__ function to implement explanation-guided occlusion """
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Locate & load feature ranking from storage
        id = path.split("/")[-1].split(".")[0]
        # 'rank_type' decides whether occlusion is signed-value or relevance dependent
        rank_pth = f"{self.rank_dir}/{id}{self.rank_type}.pkl"
        rank = torch.tensor(pickle.load(open(rank_pth, "rb"))).to(torch.int32)
        # Remove in ascending order for lowest-first, or descending for highest-first 
        rank = rank.flip(0) if self.flag_keep else rank
        rank = torch.unravel_index(rank[: self.num_remove], self.img_size[-2:])
        sample[:, *rank] = self.v_default  # Remove selected features
        return sample, target


""" 
    =============> Used for Retraining <============= 
    The implementation of the random baseline as a refernce for explainer's performance
"""
class ImageFolderRandomManip(torchvision.datasets.ImageFolder):
    def __init__(self, root, num_remove, img_size, v_default=0, random_seed=0, 
                 rank_dir=None, **kwargs):
        """
        Parameters:
        -----
        root:       [str]   Folder of data storage as in ImageFolder
        num_remove: [int]   Number of features to mask out
        img_size:   [tuple] Size of input image, in the order of (C,H,W)
        v_default:  [int]   Default value for replacement 
        random_seed:[int]   Reproducibility for fair comparison 
        rank_dir:   [str]   Folder containing explanations, only used to align the 
                            training & test samples for fair comparison; skip the 
                            filtering and use the full set if not given 
        kwargs:     standard ImageFolder arguments, e.g. transform
        """
        super().__init__(root, **kwargs)
        self.num_remove = num_remove
        self.v_default, self.img_size = v_default, img_size
        self.rank_dir = rank_dir
        # Determining the range of feature index
        self.range = torch.tensor(self.img_size)[-2:].prod()
        self.random_seed = random_seed
        self._filter_invalid_entries()

    def _filter_invalid_entries(self):
        if self.rank_dir is None:
            log(f'Use full set in {self.root} containing #entries: [{len(self.samples)}]')
            return

        samples = []
        for p in self.samples:
            path, _ = p
            id = path.split("/")[-1].split(".")[0]
            expl_pth = f'{self.rank_dir}/{id}.pkl'
            if os.path.exists(expl_pth):
                samples.append(p)
        self.samples = samples
        log(f'#Valid entries in {self.root}: [{len(self.samples)}]')

    def get_random_rank(self, index):
        """
            Using a determined and entry-specific random seed, which ensures 
            reproducibility while preserving randomness. 
            NOTE: this function produces exactly the SAME mask for a particular 
            training/test sample when visited in different epochs, which facilitates
            the training of the model by providing static observations.
        """
        torch.manual_seed(index+self.random_seed)    
        rank = torch.randperm(self.range)
        rank = torch.unravel_index(rank[: self.num_remove], self.img_size[-2:])
        return rank

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rank = self.get_random_rank(index) # generate random rank
        sample[:, *rank] = self.v_default  # Remove selected features
        return sample, target

"""
    =============> Used for Explaining <============= 
    Customization of ImageFolder, adding entry path as part of the output tuple,
    enabling tracing of unique image IDs for explanation storage
""" 
class ImageFolderReturnPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path, target


"""
    =============> Used for inference-based evaluation <============= 
    Customization of ImageFolder, used by inference-based evaluation scheme.
    Feature ranking determined by attribution score is returned as part of an entry,
    enabling convenient input perturbation without repetitively explaining.
""" 
class ImageFolderReturnRanks(torchvision.datasets.ImageFolder):
    def __init__(self, root, rank_dir, rank_type, **kwargs):
        """ Refer to ImageFolderManipulate to Args meanings """
        super().__init__(root, **kwargs)
        self.rank_dir, self.rank_type = rank_dir, rank_type

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        id = path.split("/")[-1].split(".")[0]
        rank_pth = f"{self.rank_dir}/{id}{self.rank_type}.pkl"
        rank = pickle.load(open(rank_pth, "rb"))
        return sample, path, torch.tensor(rank), target


"""
    =============> Used for inference-based evaluation <============= 
    The random baseline in inference-based evaluation 
"""
class ImageFolderRandomRanks(torchvision.datasets.ImageFolder):
    def __init__(self, root, num_remove, img_size, random_seed=0, **kwargs):
        super().__init__(root, **kwargs)
        self.num_remove = num_remove
        self.img_size = img_size
        # Determining the range of feature index
        self.range = torch.tensor(self.img_size)[-2:].prod()
        self.random_seed = random_seed

    def get_random_rank(self, index):
        torch.manual_seed(index+self.random_seed)    # Deploying an entry-specific and determined random seed that ensures randomness and reproducibility
        rank = torch.randperm(self.range)
        rank = torch.unravel_index(rank[: self.num_remove], self.img_size[-2:])
        return rank

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rank = self.get_random_rank(index) # generate random rank
        return sample, path, torch.tensor(rank), target
