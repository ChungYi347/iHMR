import random
from torch.utils.data.dataset import Dataset
import numpy as np
from .agora import AGORA
from .bedlam import BEDLAM

datasets_dict = {'bedlam': BEDLAM, 'agora': AGORA}

class MultipleDatasets(Dataset):
    def __init__(self, datasets_used, datasets_split = None, make_same_len = False, **kwargs):
        if datasets_split is None:
            self.dbs = [datasets_dict[ds](**kwargs) for ds in datasets_used]
        else:
            self.dbs = [datasets_dict[ds](split, **kwargs) for ds, split in zip(datasets_used, datasets_split)]
        
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = make_same_len
        self.human_model = self.dbs[0].human_model

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

        norm_img, meta_data = self.dbs[db_idx][data_idx]
        return norm_img, meta_data