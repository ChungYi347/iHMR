import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE
import cv2
from math import radians,sin,cos
from utils.constants import smpl_24_flip, smpl_root_idx
from utils.map import gen_scale_map, build_z_map
from utils.transforms import unNormalize, to_zorder

class PoseTrack(BASE):
    def __init__(self, split='train', **kwargs):
        super(PoseTrack, self).__init__(**kwargs)
        assert split == 'train'

        self.ds_name = 'posetrack'
        self.split = split
        self.dataset_path = os.path.join(dataset_root,'Posetrack/PoseTrack2018')
        # self.annots_path = os.path.join(self.dataset_path,'posetrack_train_smpl.npz')
        self.annots_path = os.path.join(self.dataset_path,'posetrack_train_smpl_kp2d.npz')
        self.annots = np.load(self.annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())
        
    def __len__(self):
        return len(self.img_names)
    
    def get_raw_data(self, idx):
        img_id = idx%len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path,img_name)

        pnum = len(annots)
        cam_rot = torch.eye(3,3).repeat(pnum,1,1).float()
        cam_trans = torch.zeros(pnum,3).float()

        betas_list=[]
        poses_list=[]
        transl_list=[]
        cam_intrinsics_list=[]

        ori_j2ds_list = []

        for i in range(pnum):
            #smpl and cam
            smpl_param = annots[i]['smpl_param']
            cam_param = annots[i]['cam_param']
            cam_intrinsics = torch.tensor([
                [cam_param['focal'][0], 0., cam_param['princpt'][0]],
                [0, cam_param['focal'][1], cam_param['princpt'][1]],
                [0, 0, 1]
            ])
            betas = torch.tensor(smpl_param['shape'])
            poses = torch.tensor(smpl_param['pose'])
            transl = torch.tensor(smpl_param['trans'])
            ori_j2ds = torch.tensor(smpl_param['kp2d']) 
            
            betas_list.append(betas)
            poses_list.append(poses)
            transl_list.append(transl)
            cam_intrinsics_list.append(cam_intrinsics)

            ori_j2ds_list.append(ori_j2ds)

        betas = torch.stack(betas_list).float()
        poses = torch.stack(poses_list).float()
        transl = torch.stack(transl_list).float()
        cam_intrinsics = torch.stack(cam_intrinsics_list).float()

        ori_j2ds = torch.stack(ori_j2ds_list).float()

        raw_data={'img_path': img_path,
                  'ds': 'posetrack',
                  'pnum': len(betas),
                  'betas': betas,
                  'poses': poses,
                  'transl': transl,
                  'cam_intrinsics':cam_intrinsics[0][None],
                  'cam_rot': cam_rot,
                  'cam_trans': cam_trans,
                  '3d_valid': False,
                  'detect_all_people':True,
                  'ori_j2ds': ori_j2ds, 
                    }
        
        return raw_data

    def process_data(self, img, raw_data, rot = 0., flip = False, scale = 1.):
        meta_data = copy.deepcopy(raw_data)
        # prepare rotation augmentation mat.
        rot_aug_mat = torch.tensor([[cos(radians(-rot)), -sin(radians(-rot)), 0.],
                            [sin(radians(-rot)), cos(radians(-rot)), 0.],
                            [0., 0., 1.]])
        meta_data.update({'rot_aug_mat': rot_aug_mat})

        img = self.process_img(img, meta_data, rot, flip, scale)

        self.process_cam(meta_data, rot, flip, scale)
        self.process_smpl(meta_data, rot, flip, scale)
        self.project_joints(meta_data, raw_data)
        self.check_visibility(meta_data)
        matcher_vis = meta_data['j2ds_mask'][:,:22,0].sum(dim = -1) # num of visible joints used in Hungarian Matcher
        if meta_data['pnum'] == 0 or not torch.all(matcher_vis):
            if self.mode == 'train':
                meta_data['pnum'] = 0
                return img, meta_data
 
        j3ds = meta_data['j3ds']
        depths = j3ds[:, smpl_root_idx, [2]].clone()
        if len(meta_data['cam_intrinsics']) == 1:
            focals = torch.full_like(depths, meta_data['cam_intrinsics'][0,0,0]) 
        else:
            focals = meta_data['cam_intrinsics'][:,0,0][:, None] 
        depths = torch.cat([depths, depths/focals],dim=-1)
        meta_data.update({'depths': depths, 'focals': focals})

        self.get_boxes(meta_data)
        
        meta_data.update({'labels': torch.zeros(meta_data['pnum'], dtype=int)})

        # VI. Occlusion augmentation
        if self.aug:
            occ_boxes = self.occlusion_aug(meta_data)
            for (synth_ymin, synth_h, synth_xmin, synth_w) in occ_boxes:
                img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
        
        if self.use_sat:
            # scale map
            boxes = meta_data['boxes']
            scales = boxes[:,2:].norm(p=2,dim=1).clamp(0.,1.)
            v3ds = meta_data['verts']
            depths_norm = meta_data['depths'][:,1]
            cam_intrinsics = meta_data['cam_intrinsics']
            sorted_idx = torch.argsort(depths_norm, descending=True)
            if self.backbone == "vitb":
                patch_size = 28
                map_size = (meta_data['img_size'] + 27)//28
            else:
                patch_size = 32
                map_size = (meta_data['img_size'] + 31)//32

            scale_map = gen_scale_map(scales[sorted_idx], v3ds[sorted_idx],
                                        faces = self.human_model.faces, 
                                        cam_intrinsics = cam_intrinsics[sorted_idx] if len(cam_intrinsics) > 1 else cam_intrinsics, 
                                        map_size = map_size,
                                        patch_size = patch_size,
                                        pad = True)
            scale_map_z, _, pos_y, pos_x = to_zorder(scale_map, 
                                                z_order_map = self.z_order_map,
                                                y_coords = self.y_coords,
                                                x_coords = self.x_coords)
            meta_data['scale_map'] = scale_map_z
            meta_data['scale_map_pos'] = {'pos_y': pos_y, 'pos_x': pos_x}
            meta_data['scale_map_hw'] = scale_map.shape[:2]

        return img, meta_data

    def process_smpl(self, meta_data, rot = 0., flip = False, scale = 1.):             
        poses = meta_data['poses']
        bs = poses.shape[0]
        assert poses.ndim == 2
        assert tuple(poses.shape) == (bs, self.num_poses*3) 
        # Merge rotation to smpl global_orient
        global_orient = poses[:,:3].clone()
        cam_rot = meta_data['cam_rot'].numpy()
        for i in range(global_orient.shape[0]):
            root_pose = global_orient[i].view(1, 3).numpy()
            R = cam_rot[i].reshape(3,3)
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).flatten()
            global_orient[i] = root_pose
        poses[:,:3] = global_orient
        
        # Flip smpl parameters
        if flip:
            poses = poses.reshape(bs, self.num_poses, 3)
            poses = poses[:, self.poses_flip, :]
            poses[..., 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
            poses = poses.reshape(bs, -1)

        # Update all pose params
        meta_data.update({'poses': poses})

        # Get vertices and joints in cam_coords
        with torch.no_grad():
            smpl_kwargs = {'poses': meta_data['poses'], 'betas': meta_data['betas']}
            if 'genders' in meta_data:
                smpl_kwargs.update({'genders': meta_data['genders']})
            verts, j3ds = self.human_model(**smpl_kwargs)
            # R = roma.rotvec_to_rotmat(pose[:,0])
            # poses = axis_angle_to_matrix(meta_data['poses'].reshape(-1, 24, 3))
            # smpl_out = self.smpl(global_orient = poses[:, :1],
            #                 body_pose = poses[:, 1:24],
            #                 betas = meta_data['betas'])
            # _verts, _j3ds = smpl_out.vertices, smpl_out.joints
            # import pdb; pdb.set_trace()

        j3ds = j3ds[:, :self.num_kpts, :]
        root = j3ds[:,smpl_root_idx,:].clone() # smpl root
        # new translation in cam_coords
        transl = torch.bmm((root+meta_data['transl']).reshape(-1,1,3),meta_data['cam_rot'].transpose(-1,-2)).reshape(-1,3)\
            +meta_data['cam_trans']-root
        if flip:
            transl[...,0] = -transl[...,0]

        meta_data.update({'transl': transl})

        verts = verts + transl.reshape(-1,1,3)
        j3ds = j3ds + transl.reshape(-1,1,3)
        meta_data.update({'verts': verts, 'j3ds': j3ds})

    def project_joints(self, meta_data, raw_data=None):
        j3ds = meta_data['j3ds']
        cam_intrinsics = meta_data['cam_intrinsics']
        
        # padd_cam_intrinsics = meta_data['cam_intrinsics'].clone()
        # size = meta_data['img_size'].flip(0)
        # scale = self.input_size / max(size)
        # offset = (self.input_size - scale * size) / 2
        # padd_cam_intrinsics[:,:2,-1] += offset

        j2ds_homo = torch.matmul(j3ds,cam_intrinsics.transpose(-1,-2))
        j2ds = j2ds_homo[...,:2]/(j2ds_homo[...,2,None])

        # padd_j2ds_homo = torch.matmul(j3ds,padd_cam_intrinsics.transpose(-1,-2))
        # padd_j2ds = padd_j2ds_homo[...,:2]/(padd_j2ds_homo[...,2,None])
        # meta_data.update({'j3ds': j3ds, 'j2ds': j2ds, 'padd_j2ds': padd_j2ds})
        ori_j2ds = raw_data['ori_j2ds'][:, 0, :, :2]
        ori_j2ds[:, :, 0] = ori_j2ds[:, :, 0] / max(meta_data['ori_img_size'])
        ori_j2ds[:, :, 1] = ori_j2ds[:, :, 1] / max(meta_data['ori_img_size'])
        meta_data.update({'j3ds': j3ds, 'j2ds': j2ds, 
                'ori_j2ds': raw_data['ori_j2ds'][:, 0, :, :2], 'ori_j2ds_conf': raw_data['ori_j2ds'][:, 0, :, 2]})

# class PoseTrack(BASE):
#     def __init__(self, split='train', **kwargs):
#         super(PoseTrack, self).__init__(**kwargs)
#         assert split == 'train'

#         self.ds_name = 'posetrack'
#         self.split = split
#         self.dataset_path = os.path.join(dataset_root,'Posetrack/PoseTrack2018')
#         self.annots_path = os.path.join(self.dataset_path,'posetrack_train_smpl.npz')
#         self.annots = np.load(self.annots_path, allow_pickle=True)['annots'][()]
#         self.img_names = list(self.annots.keys())
        
#     def __len__(self):
#         return len(self.img_names)
    
#     def get_raw_data(self, idx):
#         img_id = idx%len(self.img_names)
#         img_name = self.img_names[img_id]
#         annots = copy.deepcopy(self.annots[img_name])
#         img_path = os.path.join(self.dataset_path,img_name)

#         pnum = len(annots)
#         cam_rot = torch.eye(3,3).repeat(pnum,1,1).float()
#         cam_trans = torch.zeros(pnum,3).float()

#         betas_list=[]
#         poses_list=[]
#         transl_list=[]
#         cam_intrinsics_list=[]

#         for i in range(pnum):
#             #smpl and cam
#             smpl_param = annots[i]['smpl_param']
#             cam_param = annots[i]['cam_param']
#             cam_intrinsics = torch.tensor([
#                 [cam_param['focal'][0], 0., cam_param['princpt'][0]],
#                 [0, cam_param['focal'][1], cam_param['princpt'][1]],
#                 [0, 0, 1]
#             ])
#             betas = torch.tensor(smpl_param['shape'])
#             poses = torch.tensor(smpl_param['pose'])
#             transl = torch.tensor(smpl_param['trans'])
            
#             betas_list.append(betas)
#             poses_list.append(poses)
#             transl_list.append(transl)
#             cam_intrinsics_list.append(cam_intrinsics)

#         betas = torch.stack(betas_list).float()
#         poses = torch.stack(poses_list).float()
#         transl = torch.stack(transl_list).float()
#         cam_intrinsics = torch.stack(cam_intrinsics_list).float()

#         raw_data={'img_path': img_path,
#                   'ds': 'posetrack',
#                   'pnum': len(betas),
#                   'betas': betas,
#                   'poses': poses,
#                   'transl': transl,
#                   'cam_intrinsics':cam_intrinsics[0][None],
#                   'cam_rot': cam_rot,
#                   'cam_trans': cam_trans,
#                   '3d_valid': True,
#                   'detect_all_people':True
#                     }
        
#         return raw_data
