model = [
     dict(
        model_config='./models/mask-rcnn_hrnetv2p-w32-2x_FAST.py',
        checkpoint='./models/hrnet_epoch_36.pth',
        device='cuda:0'),
]

preprocess = dict(
    downsample_time=16,
    downsample_freq=16,
    freq_range=(1000, 1500),
    tbox=50,
    basebandStd=1.0,
    scaling=0.8,
    nsubint=4  
)
postprocess = dict(
    threshold=0.10,
    nms_cfg=dict(iou_threshold=0.85, max_candidates=20),
    mapping=dict(
        ransac_cfg=dict(sample_points=100, iterations=15, fit_pair=True)),
    aug_cfg=dict(type='dm_filtering', threshold=3.0))
unpack_2bit = True
