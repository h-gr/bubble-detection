_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
data_root = 'data/coco/'
classes = ('bubble', )
data = dict(
    train=dict(classes=classes,
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',),
    val=dict(classes=classes,
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',),
    test=dict(classes=classes,
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',))
