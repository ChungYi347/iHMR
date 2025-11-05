import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class PW3D(BASE):
    def __init__(self, split='train', target_folders=None, **kwargs):
        super(PW3D, self).__init__(**kwargs)
        assert split in ['train','test']

        self.ds_name = '3dpw'
        self.split = split
        self.dataset_path = os.path.join(dataset_root,'3dpw')
        annots_path = os.path.join(self.dataset_path,'annots_smpl_{}_genders.npz'.format(split))
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())

        if target_folders is not None:
            self.target_folders = target_folders
            filtered = []
            # input_dir
            # pw3d_folder_path = "./data/datasets/3dpw/imageFiles/" + target_folders.split('/')[-1]
            # files = [target_folders.split('/')[-1] + "/" + img_file_path for img_file_path in os.listdir(target_folders)]
            # max_num = max([int(img_file_path.split('/')[-1].replace('image_', '').replace('.jpg',  '')) for img_file_path in os.listdir(target_folders)])
            for name in self.img_names:
                if name.split('/')[1] in target_folders:
                    filtered.append(name)
                    # # import pdb; pdb.set_trace()
                    # if int(name.split('/')[-1].replace('image_', '').replace('.jpg',  '')) <= max_num:
                    #     filtered.append(name)
                    # # if name.replace('imageFiles/', '') in files:
                    # #     filtered.append(name)
            self.img_names = filtered
            self.img_names.sort()
            print(f"[PW3D] Using {len(self.img_names)} images from folders: {target_folders}")
        
    def __len__(self):
        return len(self.img_names)
    
    def get_raw_data(self, idx):
        img_id = idx%len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path,img_name)

        pnum = len(annots['betas'])

        cam_intrinsics = torch.tensor(annots['cam_intrinsics']).float().unsqueeze(0)
        cam_rot = torch.tensor(annots['cam_rot']).repeat(pnum,1,1).float()
        cam_trans = torch.tensor(annots['cam_trans']).repeat(pnum,1).float()
        
        betas = annots['betas']
        poses = torch.cat([annots['global_orient'], annots['body_pose']], dim=1)
        transl = annots['transl']

        genders = annots['genders'] if 'genders' in annots else annots['gender']
        genders = ['female' if gender.lower() in ['f', 'female'] else 'male' for gender in genders]

        raw_data={'img_path': img_path,
                  'ds': '3dpw',
                  'pnum': len(betas),
                  'betas': betas,
                  'poses': poses,
                  'transl': transl,
                  'cam_rot': cam_rot,
                  'cam_trans': cam_trans,
                  'cam_intrinsics':cam_intrinsics,
                  '3d_valid': True,
                  'genders': genders,
                  'detect_all_people':False
                    }
        
        return raw_data


