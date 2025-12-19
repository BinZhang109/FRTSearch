model = [
     dict(
        model_config='/apdcephfs/private_binhezhang/FRTSearch/models/mask-rcnn_hrnetv2p-w32-2x_FAST.py',
        checkpoint='/apdcephfs/private_binhezhang/FRTSearch/models/epoch_36.pth',
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
    threshold=0.05,
    nms_cfg=dict(iou_threshold=0.85, max_candidates=20),
    mapping=dict(
        ransac_cfg=dict(sample_points=40, iterations=10, fit_pair=True)),
    aug_cfg=dict(type='dm_filtering', threshold=3.0))
unpack_2bit = True
