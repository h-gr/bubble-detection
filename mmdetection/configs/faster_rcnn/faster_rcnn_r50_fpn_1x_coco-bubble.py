_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
data_root = 'data/'
classes = ('bubble', )
data = dict(
    train=dict(classes=classes,
    ann_file=data_root + 'train/_annotations.coco.json',
    img_prefix=data_root + 'train/',),
    val=dict(classes=classes,
    ann_file=data_root + 'valid/_annotations.coco.json',
    img_prefix=data_root + 'valid/',),
    test=dict(classes=classes,
    ann_file=data_root + 'test/_annotations.coco.json',
    img_prefix=data_root + 'test/',))
