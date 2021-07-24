_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
	type='FasterRCNND',
    backbone=dict(
        conv_cfg = dict(type='ConvAWS'),
        ),
    neck=dict(
            type='NFPN',
            in_channels=256,
            out_channels=256,
            num_levels=5,
            conv_cfg = dict(type='ConvAWS'),
            norm_cfg=dict(type='BN', requires_grad=True),
            num_outs=5),
    roi_head=dict(
        type='StandardRoIHeadD',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractorD',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]))
    )