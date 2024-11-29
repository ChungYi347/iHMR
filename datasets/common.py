import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from configs.paths import dataset_root
import copy
from .base import BASE

# dataset for inference
class COMMON(BASE):
    def __init__(self, img_folder, **kwargs):
        super(COMMON, self).__init__(**kwargs)
        self.dataset_path = img_folder
        self.img_names = sorted([img_name\
                                 for img_name\
                                 in os.listdir(self.dataset_path)\
                                 if img_name.endswith('.png') or img_name.endswith('.jpg')  or img_name.endswith('.jpeg')])
        assert self.mode == 'infer'
        
    def __len__(self):
        return len(self.img_names)
    
    def get_raw_data(self, idx):
        img_id=idx%len(self.img_names)
        img_name=self.img_names[img_id]
        img_path=os.path.join(self.dataset_path,img_name)
        raw_data={'img_path': img_path,
                'img_name': img_name,
                'ds': 'common'
                    }
        
        return raw_data


