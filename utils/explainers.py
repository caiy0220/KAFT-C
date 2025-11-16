import numpy as np
import torch
from torchvision.transforms import GaussianBlur
from utils import basic

class GradBased:
    def _get_grad(self, m, img, lbl):
        img = img.requires_grad_(True)
        output = m(img)[:, lbl].sum()
        # return torch.autograd.grad(torch.unbind(output), img)[0].cpu()
        return torch.autograd.grad(output, img)[0].cpu()

class SG(GradBased):
    def __init__(self, num=50, sigma=0.25):
        super(SG, self).__init__()
        self.num, self.sigma = max(num, 1), sigma
        self.pv_floor, self.pv_ceil = 0., 1.
    
    def set_pixel_value_bounds(self, pv_floor, pv_ceil):
        self.pv_floor, self.pv_ceil = pv_floor, pv_ceil

    def explain(self, m, img, batch_size=64, lbl=None):
        device = basic.get_device(m)
        lbl = np.argmax(m(img.unsqueeze(0).to(device))[0].detach().cpu()) if lbl is None else lbl
        grad = self._get_grad(m, img.unsqueeze(0).to(device), lbl)[0] / self.num

        if self.num-1 > 0:
            noises = torch.randn((self.num,) + img.shape[-3:]) * self.sigma
            noised = noises + img
        for ptr in range(0, self.num-1, batch_size):
            batch = noised[ptr:ptr+batch_size]
            batch_grads = self._get_grad(m, batch.to(device), lbl)
            batch_contribution = torch.sum(batch_grads, dim=0) / self.num
            grad += batch_contribution
        return grad
    
class VG(SG):
    def __init__(self):
        super().__init__(1, 0.)
 
class IG(GradBased):
    def __init__(self, steps, input_size, baseline=None):
        super(IG, self).__init__()
        self.baseline = baseline if baseline is not None else torch.zeros(input_size)
        self.steps = max(steps, 1)
        self.smooth_num, self.smooth_sigma = 1, 0.25
        self.aligned_noise = False
        self.blur_m = GaussianBlur((31, 31), 5.0)

    def explain(self, m, img, batch_size=64, lbl=None):
        """ Take a single image as input in shape [C x H x W] """
        device = basic.get_device(m)
        lbl = np.argmax(m(img.unsqueeze(0).to(device))[0].detach().cpu()) if lbl is None else lbl
        baseline = self._get_baseline(img)
        grad = torch.zeros(img.shape[-3:])

        samples = self._get_path(img, baseline)
        for ptr in range(0, len(samples), batch_size):
            batch = samples[ptr:ptr+batch_size]
            batch_grads = self._get_grad(m, batch.to(device), lbl)
            batch_contribution = torch.sum(batch_grads, dim=0) / len(samples)
            grad += batch_contribution
        
        dx = img - baseline
        grad = grad * dx[0] / self.steps
        return grad 
    
    def _get_path(self, img, baseline):
        paths = [baseline + (float(i)/self.steps) * (img - baseline) for i in range(self.steps+1)]
        paths = torch.stack(paths)
        noise_num = self.smooth_num - 1
        if noise_num > 0:
            noises = torch.randn((noise_num,) + paths.shape) * self.smooth_sigma
            noised = noises + paths
            paths = torch.concat([noised.flatten(0, 1), paths])
        return paths

    def _get_baseline(self, x):
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':   
            # explicand-specific baseline, requiring definition of 'self.blur_m', only apply to IMG
            if self.blur_m is None: 
                basic.log('Bluring kernel is not defined, use default', lvl=0)
                self.set_bluring_kernel()
            baselines = self.blur_m(x)
        elif isinstance(self.baseline, (int, float)):
            # Constant value baseline, identical value across different channels (if applies)
            baselines = self.baseline
        elif isinstance(self.baseline, torch.Tensor):
            # Image-like baseline, having shape [C*H*W]
            baselines = self.baseline
        elif isinstance(self.baseline, tuple):
            # Constant value baseline for multi-channel image only
            baselines = torch.Tensor(self.baseline).reshape(len(self.baseline),1,1)
        else:
            assert 1 == 0, 'Unsupported baseline type'
        return baselines

    """ 
    ============ IG + SG ============
    """
    def set_smooth(self, num: int, sigma=0.25):
        self.smooth_num = num
        self.smooth_sigma = sigma

class SIG(IG):
    def __init__(self, steps, input_size, baseline=None, sigma=0.25):
        super().__init__(steps, input_size, baseline)
        self.smooth_sigma = sigma

    def _get_path(self, img, baseline):
        paths = [baseline + (float(i)/self.steps) * (img - baseline) for i in range(self.steps+1)]
        paths = torch.stack(paths)
        noises = torch.randn_like(paths) * self.smooth_sigma
        return paths + noises
