# 3D Self-Supervision by Distilling 2D Open-Vocabulary Segmentation Models for Annotation-free Training

## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](DATASETS.md)

## Sueprpixel-superpoint generation

You can choose from the following three open-vocabulary segmentation models:

For CAT-Seg:
Please prepare the checkpoint and other related content according to [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg/tree/main).
```
# superpixels generation
cd ov_segment/CAT-Seg
python demo/superpixel_generation.py
# superpoints generation
python superpixel2superpoint.py
```

For FC-CLIP:
Please prepare the checkpoint and other related content according to [FC-CLIP](https://github.com/bytedance/fc-clip/tree/main).
```
# superpixels generation
cd ov_segment/fc-clip
python superpixel_generation.py
# superpoints generation
python superpixel2superpoint.py
```

For SAN:
Please prepare the checkpoint and other related content according to [SAN](https://github.com/MendelXu/SAN/tree/main).
```
# superpixels generation
cd ov_segment/SAN
python superpixel_generation.py
# superpoints generation
python superpixel2superpoint.py
```

## Training
### Prertrain:
For example:
```
# pretrain
python pretrain.py --cfg_file "config/pretrain_san.yaml"
```

### Annotation-free training

For example:
```
# annotation-free
python annotation_free.py --cfg_file "config/annotation_free_san.yaml" --pretraining_path "SSOV_pretrain_san.pt"
```
We use --pretraining_path ["minkunet_slidr.pt"](https://github.com/valeoai/SLidR) as baseline.

Or, you can change ```$training : 'parametrize'``` in config/annotation_free_XXX.yaml for pretraining.


### Finetuning
For example:
```
# finetune (SSOV-TMP)
python downstream.py --cfg_file "config/semseg_nuscenes.yaml" --pretraining_path "SSOV_pretrain_san.pt"
# or (SSOV)
python downstream.py --cfg_file "config/semseg_nuscenes.yaml" --pretraining_path "SSOV_pretrain_with_af_san.pt"
# SemanticKITTI
python downstream.py --cfg_file "config/semseg_kitti.yaml" --pretraining_path "SSOV_pretrain_with_af_san.pt"

Or, you can change:
```
dataset_skip_step : 1
freeze_layers : True
lr : 0.05
lr_head : Null
```
in config/semseg_nuscenes.yaml for linear probing.

### Evaluation
For example:
```
# evaluate
python evaluate.py --cfg_file "config/annotation_free_san.yaml" --resume_path "SSOV_af_with_pretrain_san.pt"
```


