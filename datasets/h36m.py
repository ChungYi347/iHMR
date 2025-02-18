import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from configs.paths import dataset_root
import copy
from .base import BASE

class H36M(BASE):
    def __init__(self, split='train', **kwargs):
        super(H36M, self).__init__(**kwargs)
        assert split == 'train'
        self.ds_name = 'h36m'
        self.split = split
        self.dataset_path = os.path.join(dataset_root,'h36m')
        annots_path = os.path.join(self.dataset_path,'annots_smpl_train_small.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())
        
    def __len__(self):
        return len(self.img_names)
    
    def get_raw_data(self, idx):
        img_id = idx%len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path,img_name)

        cam_intrinsics = torch.from_numpy(annots['cam_intrinsics']).float().unsqueeze(0)
        cam_rot = torch.from_numpy(annots['cam_rot']).float().unsqueeze(0)
        cam_trans = torch.from_numpy(annots['cam_trans']).float().unsqueeze(0)
        
        betas = annots['betas']
        poses = torch.cat([annots['global_orient'].flatten(1), annots['body_pose'].flatten(1)], dim=1)
        transl = annots['transl']

        raw_data={'img_path': img_path,
                  'ds': 'h36m',
                'pnum': len(betas),
                'betas': betas,
                'poses': poses,
                'transl': transl,
                'cam_rot': cam_rot,
                'cam_trans': cam_trans,
                'cam_intrinsics':cam_intrinsics,
                '3d_valid': True,
                'detect_all_people':True
                    }
        
        return raw_data