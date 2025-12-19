
## 以单个文件为单位，输出_txt(参数列表)，输出_dyn(候选图)
##### 绘制诊断图
python FRTSearch.py  /apdcephfs/private_binhezhang/deployment_zhang/datasource/Dec-0221_arcdrift-1_20200812/B1911-04_Dec-0452_arcdrift-M18_1384_2bit.fits ./configs/detector_FAST_19beams.py --slide 128
+ 输出结果：该文件下面B1911-04_Dec-0452_arcdrift-M18_1384_2bit_results目录
+ plots 诊断图
+ B1911-04_Dec-0452_arcdrift-M18_1384_2bit_candidates.txt 参数

##### 不绘制诊断图
python FRTSearch.py  /apdcephfs/private_binhezhang/deployment_zhang/datasource/Dec-0221_arcdrift-1_20200812/B1911-04_Dec-0452_arcdrift-M18_1384_2bit.fits ./configs/detector_FAST_19beams.py --slide 128 --no-plots