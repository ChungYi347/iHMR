import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from configs.paths import dataset_root
import copy
from tqdm import tqdm
from .base import BASE

class BEDLAM(BASE):
    def __init__(self, split='train_6fps',**kwargs):
        super(BEDLAM, self).__init__(**kwargs)
        assert split in ['train_1fps','train_3fps','train_6fps','validation_6fps']
        assert not self.kid_offset

        self.ds_name = 'bedlam'
        self.dataset_path = os.path.join(dataset_root,'bedlam')
        annots_path = os.path.join(self.dataset_path,f'bedlam_smpl_{split}.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())
        self.split = 'train' if 'train' in split else 'validation'
        
    def __len__(self):
        return len(self.img_names)

    def cnt_instances(self):
        ins_cnt = 0
        for idx in tqdm(range(len(self))):
            img_id = idx
            img_name = self.img_names[img_id]
            # ins_cnt += len(self.annots[img_name]['isValid'])
            ins_cnt += len(self.annots[img_name]['shape'])
            # tqdm.write(str(ins_cnt))

        print(f'TOTAL: {ins_cnt}')
    
    def get_raw_data(self, idx):

        img_id = idx%len(self.img_names)
        img_name = self.img_names[img_id]
        
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path,self.split,img_name)

        cam_intrinsics = torch.from_numpy(annots['cam_int']).unsqueeze(0)
        cam_rot = torch.from_numpy(np.stack(annots['cam_rot']))
        cam_trans = torch.from_numpy(np.stack(annots['cam_trans']))
        
        betas = torch.from_numpy(np.stack(annots['shape']))
        poses = torch.from_numpy(np.stack(annots['pose_world']))
        transl = torch.from_numpy(np.stack(annots['trans_world']))

        raw_data={'img_path': img_path,
                'ds': 'bedlam',
                'pnum': len(betas),
                'betas': betas.float(),
                'poses': poses.float(),
                'transl': transl.float(),
                'cam_rot': cam_rot.float(),
                'cam_trans': cam_trans.float(),
                'cam_intrinsics':cam_intrinsics.float(),
                '3d_valid': True,
                'age_valid': False,
                'detect_all_people':True
                    }

        if self.mode == 'eval':
            raw_data['occ_level'] = torch.zeros(len(betas),dtype=int)
        
        return raw_data


