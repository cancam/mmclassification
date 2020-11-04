_base_ = [
    '../_base_/models/resnet152_coco.py', '../_base_/datasets/coco_bs64_augm.py',
    '../_base_/schedules/coco_bs256.py', '../_base_/default_runtime.py'
]
