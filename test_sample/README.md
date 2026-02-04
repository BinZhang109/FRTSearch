# FRTSearch Test Samples

This notebook evaluates FRTSearch detection performance and parameter configurations using public FRB datasets from **FAST** (<a href="https://doi.org/10.3847/1538-4365/adf42d">Guo et al. 2025</a>) and **SKA** (<a href="https://doi.org/10.1038/s41586-018-0588-y">Shannon et al. 2018</a>). Additional samples from **Parkes** (<a href="https://doi.org/10.1093/mnras/stu2650">Keane et al. 2015</a>) and **SKA** (<a href="https://doi.org/10.1126/science.adf2678">Ryder et al. 2022</a>) are included to demonstrate hyperparameter tuning across diverse telescope specifications.

---

## Part 1: Test Dataset Overview

<table>
    <thead>
        <tr>
            <th>Source Name</th>
            <th>Telescope</th>
            <th>Reference</th>
            <th>Sampling Time (μs)</th>
            <th>Freq. Range (MHz)</th>
            <th>Channels</th>
            <th>DM (pc cm<sup>-3</sup>)</th>
            <th>ToA (s)</th>
            <th>bits</th>
        </tr>
    </thead>
    <tbody>
        <!-- FAST Group -->
        <tr>
            <td>FRB20121102</td>
            <td rowspan="3" style="vertical-align: middle; text-align: center;">FAST</td>
            <td rowspan="3" style="vertical-align: middle; text-align: center;"><a href="https://doi.org/10.3847/1538-4365/adf42d">Guo et al. (2025)</a></td>
            <td>98.304</td>
            <td>1000-1500</td>
            <td>4096</td>
            <td>565.0</td>
            <td>2.39</td>
            <td>8</td>
        </tr>
        <tr>
            <td>FRB20180301</td>
            <td>49.152</td>
            <td>1000-1500</td>
            <td>4096</td>
            <td>420.0</td>
            <td>2.83</td>
            <td>8</td>
        </tr>
        <tr>
            <td>FRB20201124</td>
            <td>49.152</td>
            <td>1000-1500</td>
            <td>4096</td>
            <td>525.0</td>
            <td>1.11</td>
            <td>8</td>
        </tr>
        <!-- SKA Group 1 -->
        <tr>
            <td>FRB20180119</td>
            <td rowspan="2" style="vertical-align: middle; text-align: center;">SKA</td>
            <td rowspan="2" style="vertical-align: middle; text-align: center;"><a href="https://doi.org/10.1038/s41586-018-0588-y">Shannon et al. (2018)</a></td>
            <td>1266.46875</td>
            <td>1130-1465</td>
            <td>336</td>
            <td>402.0</td>
            <td>1679.6</td>
            <td>8</td>
        </tr>
        <tr>
            <td>FRB20180212</td>
            <td>1266.46875</td>
            <td>1130-1465</td>
            <td>336</td>
            <td>167.0</td>
            <td>1848</td>
            <td>8</td>
        </tr>
        <!-- Parkes -->
        <tr>
            <td>FRB20110220</td>
            <td style="text-align: center;">Parkes</td>
            <td style="text-align: center;"><a href="https://doi.org/10.1093/mnras/stu2650">Keane et al. (2015)</a></td>
            <td>64</td>
            <td>1182-1581</td>
            <td>1024</td>
            <td>944.0</td>
            <td>209</td>
            <td>2</td>
        </tr>
        <!-- SKA Group 2 -->
        <tr>
            <td>FRB20220610A</td>
            <td style="text-align: center;">SKA</td>
            <td style="text-align: center;"><a href="https://doi.org/10.1126/science.adf2678">Ryder et al. (2022)</a></td>
            <td>1182.09375</td>
            <td>1104-1439</td>
            <td>336</td>
            <td>1458.0</td>
            <td>1298</td>
            <td>32</td>
        </tr>
    </tbody>
</table>

---

## Part 2: Configuration & Hyperparameter Tuning

### 2.1 Parameter Structure

The configuration is divided into model loading, preprocessing, and post-processing.

```python
model = [dict(
    model_config='./models/mask-rcnn_hrnetv2p-w32-2x_FAST.py', # Network config path
    checkpoint='./models/hrnet_epoch_36.pth',                  # Model weights
    device='cuda:0')]                                          # GPU device

preprocess = dict(
    downsample_time=16,      # Time downsampling factor
    downsample_freq=16,      # Frequency downsampling factor
    freq_range=(1000, 1500), # Observation frequency range (MHz)
    tbox=50,                 # Time width for de-dispersed dynamic spectra plots
    basebandStd=1.0,         # Std-dev multiplier for RFI masking
    scaling=0.8,             # Normalization factor
    nsubint=4)               # Number of subints for waterfall plots

postprocess = dict(
    threshold=0.10,          # Confidence threshold
    nms_cfg=dict(iou_threshold=0.85, max_candidates=20), # NMS for overlapping masks
    mapping=dict(ransac_cfg=dict(sample_points=100, iterations=15, fit_pair=True)), # IMPIC fitting
    aug_cfg=dict(type='dm_filtering', threshold=3.0))    # Filter candidates with DM < 3.0
```

**CLI Argument:**
*   `--slide-size`: Number of sub-integrations (subints) loaded per batch.

### 2.2 Tuning Guide

#### Model Input Constraint

The model is trained on input dimensions of **$256 \text{ (freq)} \times 8192 \text{ (time)}$**. This wide training range enables FRTSearch to handle both drift scan short-term observations and tracking long-term observation data processing.

**Key Points:**
*   Standard input: 1 subint = 1024 time samples
*   Sliding window required when observation duration > 8192 samples
*   Goal: Adjust `downsample_time`, `downsample_freq`, and `--slide-size` to match model dimensions

#### Telescope-Specific Strategies

| Parameter | **SKA** (Low Channel Count) | **Parkes** (High Channel, Low Bit-depth) |
| :--- | :--- | :--- |
| **Characteristics** | ~3000s duration, **336 channels**. | >1000s duration, **1024 channels**, 2-bit (weak signal). |
| **Strategy** | Channels are close to 256; no freq downsampling needed. | Downsample freq to match model; downsample time to boost SNR. |
| **downsample_freq** | **1** | **4** (Reduces 1024 $\to$ 256) |
| **downsample_time** | **1** | **16** |
| **--slide-size** | **8** | **128** |
| **Calculation** | $8 \times 1024 / 1 = 8192$ | $128 \times 1024 / 16 = 8192$ |

#### Critical Rules

> **⚠️ IMPORTANT CONSTRAINTS:**
> 
> 1.  **Time Dimension:** After downsampling, window size must be **≤ 8192**
> 
> 2.  **Frequency Dimension:** Target **≈ 256** bins (can be higher like 336, but avoid very low values like 16)
> 
> 3.  **Frequency Range:** Always update `freq_range` to match your observation data

---

## Part 3: Detection Results

All test results are available in `test_samples.ipynb`. Each detection output includes:
- Detected FRB parameters (ToA, DM,  confidence)
- Processing time breakdown by stage
- Diagnostic plots for one candidate

Example output:

![alt text](image.png)
