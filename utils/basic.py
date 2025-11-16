import pickle, yaml
import os, random, argparse
import numpy as np
import torch

def get_device(m):
    return next(m.parameters()).device

def pkl_dump_wb(fn, content):
    with open(fn, 'wb') as f:
        pickle.dump(content, f)

'''
==================================================================
|                         Logging methods                        |
==================================================================
'''
import time

LOG_LEVEL = 1
# 0: all log allowed
# 1: debug disabled
# 2: only warning and error
# 3: only error
LOG_PREF = ['DEBUG', 'INFO', 'WARN', 'ERR']
MAX_LINE_WIDTH = 80

def log(content='', lvl=1, end='\n'):
    if LOG_LEVEL > lvl:
        return

    pref = LOG_PREF[lvl] + '[' + time.strftime('%x,%X') + ']:'
    print(pref, end='\t')
    print(content, end=end)

def set_log_level(lvl):
    global LOG_LEVEL
    LOG_LEVEL = lvl

def heading(msg):
    remains = MAX_LINE_WIDTH - len(msg) - 2
    return '|' + ' ' * (remains // 2) + msg + ' ' * (remains // 2 + remains % 2) + '|'

def load_config(path):
    with open(path, 'r') as f:
        return yaml.full_load(f)

def parse_config(config_pth):
    config = load_config(config_pth)
    parser = argparse.ArgumentParser()
    ns = argparse.Namespace()
    ns.__dict__.update(config)
    for k in ns.__dict__:
        parser.add_argument('--' + k, type=type(getattr(ns, k)))
    return parser, ns

def attr_quantization(x):
    x = x / x.abs().max() * 10000
    return x.to(torch.int16)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
