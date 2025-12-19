work_dir = '/data1/yukiyxli/fast_process/dev/'
# model = [
#     dict(
#         model_config='/data/service/deployment_seg/models/maskrcnn.py',
#         checkpoint='/data/service/deployment_seg/models/maskrcnn.pth',
#         device='cuda:0'),
#     dict(
#         model_config='/data/service/deployment_seg/models/cascaded_maskrcnn.py',
#         checkpoint='/data/service/deployment_seg/models/cascaded_maskrcnn.pth',
#         device='cuda:0')
# ]
model = [
    # dict(
    #     model_config='/data/service/deployment_seg/models_zhang/maskrcnn.py',
    #     checkpoint='/data/service/deployment_seg/models_zhang/maskrcnn_250319.pth',
    #     device='cuda:0'),
    # dict(
    #     model_config='/data/service/deployment_seg/models_zhang/cascade-mask-rcnn_hrnetv2p-w32_20e_coco_FAST.py',
    #     checkpoint='/data/service/deployment_seg/models_zhang/epoch_24.pth',
    #     device='cuda:0')
    # dict(
    #     model_config='/apdcephfs/private_binhezhang/deployment_zhang/models/convnext.py',
    #     checkpoint='/apdcephfs/private_binhezhang/deployment_zhang/models/convnext.pth',
    #     device='cuda:0'),
    dict(
        model_config='/apdcephfs/private_binhezhang/deployment_zhang/all_873_anno_20251115/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_FAST_25-11-15-10-52-07/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_FAST.py',
        checkpoint='/apdcephfs/private_binhezhang/deployment_zhang/all_873_anno_20251115/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_FAST_25-11-15-10-52-07/epoch_36.pth',
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
    duration=2  # 取几个subint
    )
postprocess = dict(
    threshold=0.05,
    nms_cfg=dict(iou_threshold=0.85, max_candidates=20),
    mapping=dict(
        ransac_cfg=dict(sample_points=40, iterations=10, fit_pair=True,visualize=False,slide_start_time=0)),
    aug_cfg=dict(type='dm_filtering', threshold=20.0))
unpack_2bit = True
