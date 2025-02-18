from datasets.multiple_datasets import datasets_dict     

ds_names = ['agora','bedlam','crowdpose','mpii','coco','3dpw','h36m']
ds_splits =  ['train','train_6fps','train','train','train','train','train']

# use it to visualize GTs
if __name__ == '__main__':
    kwargs = {'input_size': 1288, 'aug': False, 'mode': 'train', 'human_type':'smpl'}
    for name, split in zip(ds_names, ds_splits):
        kwargs['sat_cfg'] = {'use_sat': True, 'num_lvls':3}
        print(f'Loading {name}_{split}...')
        ds = datasets_dict[name](split = split, **kwargs)
        print(f'Length of {name}_{split}: {len(ds)}')

        ds.visualize(vis_num = 20)
    