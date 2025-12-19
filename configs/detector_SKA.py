model = [
     dict(
        model_config='./models/mask-rcnn_hrnetv2p-w32-2x_FAST.py',
        checkpoint='./models/hrnet_epoch_36.pth',
        device='cuda:0'),
]

preprocess = dict(
    downsample_time=1,
    downsample_freq=1,
    freq_range=(1130, 1465),
    tbox=20,
    basebandStd=1.0,
    scaling=0.8,
    nsubint=6,
    duration=2 
    )
postprocess = dict(
    threshold=0.10,
    nms_cfg=dict(iou_threshold=0.85, max_candidates=20),
    mapping=dict(
        ransac_cfg=dict(sample_points=100, iterations=15, fit_pair=True)),
    aug_cfg=dict(type='dm_filtering', threshold=3.0))
unpack_2bit = True
