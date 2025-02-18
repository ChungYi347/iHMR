# Data Preparation

Please prepare datasets and our preprocessed annotations following instructions below. 

* Select a folder on your device to store all datasets and annotations. Then specify this folder in `${Project}/configs/path.py: L1`.

* Download [AGORA](https://agora.is.tue.mpg.de/index.html), [BEDLAM](https://bedlam.is.tue.mpg.de/index.html), [COCO](https://cocodataset.org/#home), [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset), [CrowdPose](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view), [H36M](http://vision.imar.ro/human3.6m/description.php) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) from their official websites. 
    * For [AGORA](https://agora.is.tue.mpg.de/index.html), we use the 1280x720 images.
    * For [BEDLAM](https://bedlam.is.tue.mpg.de/index.html), we use 6fps version.
    * For [COCO](https://cocodataset.org/#home), we use 2017 train images.

* Download preprocessed annotations from [Google drive](https://drive.google.com/drive/folders/1aIr8L1gWuPSfJRNNf-YVh1Upb_agubq4?usp=sharing). We refit kid samples in [AGORA](https://agora.is.tue.mpg.de/index.html) using adult SMPL model since SAT-HMR won't predict age offsets for now.

The directory structure should be like this.

```
${dataset_root}
|-- 3dpw
    |-- imageFiles
    |-- annots_smpl_test_genders.npz
    `-- annots_smpl_train_genders.npz
|-- agora
    |-- smpl_neutral_annots
        |-- annots_smpl_train_fit.npz
        `-- annots_smpl_validation.npz
    |-- test
    |-- train
    `-- validation
|-- bedlam
    |-- train
    |-- validation
    |-- bedlam_smpl_train_1fps.npz
    |-- bedlam_smpl_train_6fps.npz
    `-- bedlam_smpl_validation_6fps.npz
|-- coco
    |-- train2017
    `-- COCO_small_NA_SMPL.npz
|-- crowdpose
    |-- images
    `-- CP_NA_SMPL_train.npz
|-- h36m
    |-- images
    `-- annots_smpl_train_small.npz
`-- mpii
    |-- images
    `-- MPII_NA_SMPL.npz
```