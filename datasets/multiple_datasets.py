import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset
import numpy as np
from .agora import AGORA
from .bedlam import BEDLAM
from .crowdpose import CROWDPOSE
from .mpii import MPII
from .coco import COCO
from .pw3d import PW3D
from .h36m import H36M
from .posetrack import PoseTrack
from .pw3d_video import PW3DVideo
from .posetrack_video import PoseTrackVideo
from .bedlam_video import BEDLAMVideo

datasets_dict = {'bedlam': BEDLAM, 'agora': AGORA, 
                 'crowdpose': CROWDPOSE, 'mpii': MPII, 
                 'coco': COCO, '3dpw': PW3D, 'h36m': H36M,
                 'posetrack': PoseTrack,
                 '3dpw_video': PW3DVideo, 'posetrack_video': PoseTrackVideo,
                 'bedlam_video': BEDLAMVideo}

class MultipleDatasets(Dataset):
    # def __init__(self, datasets_used, datasets_split = None, make_same_len = False, **kwargs):
    #     if datasets_split is None:
    #         self.dbs = [datasets_dict[ds](**kwargs) for ds in datasets_used]
    #     else:
    #         self.dbs = [datasets_dict[ds](split, **kwargs) for ds, split in zip(datasets_used, datasets_split)]
        
    #     self.db_num = len(self.dbs)
    #     self.max_db_data_num = max([len(db) for db in self.dbs])
    #     self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
    #     self.make_same_len = make_same_len
    #     self.human_model = self.dbs[0].human_model
    def __init__(self, datasets_used, datasets_split=None, make_same_len=False,
                 sampling_weights=None, **kwargs):
        if datasets_split is None:
            self.dbs = [datasets_dict[ds](**kwargs) for ds in datasets_used]
        else:
            self.dbs = [datasets_dict[ds](split, **kwargs) for ds, split in zip(datasets_used, datasets_split)]

        self.db_num = len(self.dbs)
        self.human_model = getattr(self.dbs[0], "human_model", None)

        orig_lens = np.array([len(db) for db in self.dbs], dtype=np.int64)

        if sampling_weights is not None:
            weights = np.array(sampling_weights, dtype=float)
            assert len(weights) == self.db_num
            raw_targets = orig_lens * weights
            target_lengths = np.floor(raw_targets).astype(int)
        else:
            target_lengths = orig_lens.copy()

        new_dbs = []
        rng = np.random.default_rng()
        for db, tgt_len, orig_len in zip(self.dbs, target_lengths.tolist(), orig_lens.tolist()):
            if tgt_len <= orig_len:
                idxs = rng.choice(orig_len, size=tgt_len, replace=False).tolist()
            else:
                idxs = rng.choice(orig_len, size=tgt_len, replace=True).tolist()
            new_dbs.append(Subset(db, idxs))
        self.dbs = new_dbs

        lens = [len(db) for db in self.dbs]
        print([db.dataset for db in self.dbs])
        self.max_db_data_num = max(lens)
        self.db_len_cumsum = np.cumsum(lens)
        print(f"Original lengths: {orig_lens.tolist()}")
        print(f"Target lengths  : {lens}")
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        # norm_img, meta_data, mask = self.dbs[db_idx][data_idx]
        norm_img, meta_data = self.dbs[db_idx][data_idx]
        return norm_img, meta_data