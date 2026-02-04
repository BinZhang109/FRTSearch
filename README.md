<div align="center">
  <img src="logo2.svg" width="100%" alt="FRTSearch Banner">
</div>

<!-- <h1 align='center'> FRTSearch </h1> -->

<div align="center">

🚀 **Unified Detection and Parameter Inference of Fast Radio Transients using Instance Segmentation** 🚀

<!-- Badges on one line -->
[![Paper](https://img.shields.io/badge/Paper-AASTeX-blue.svg)](https://doi.org/10.57760/sciencedb.Fastro.00038) [![Dataset](https://img.shields.io/badge/Dataset-CRAFTS--FRT-yellow.svg)](https://doi.org/10.57760/sciencedb.Fastro.00038) [![Framework](https://img.shields.io/badge/Framework-MMDetection-red.svg)](https://github.com/open-mmlab/mmdetection) [![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-grey.svg)](./LICENSE)

<!-- [Description](#description) •
[Key Features](#key-features) •
[Installation](#installation) •
[Usage](#usage) •
[Datasets](#datasets) •
[Performance](#performance) •
[Citation](#citation) -->

</div>

## Description

**FRTSearch** is an end-to-end framework that unifies the detection and physical characterization of Fast Radio Transients (FRTs), including FRBs, Pulsars, and RRATs.

Motivated by the morphological universality of dispersive trajectories in time-frequency dynamic spectra ($t \propto \nu^{-2}$), our approach reframes FRT detection as a pattern recognition problem. 

### 🔬 Method Overview

<div align="center">
  <img src="pipeline.png" width="190%" alt="FRTSearch Pipeline">
  <!-- <p><i>Figure: End-to-end pipeline of FRTSearch framework, from raw observation data to physical parameter inference.</i></p> -->
</div>

The pipeline consists of two core components:
1.  **Mask R-CNN Detection:** A deep learning model trained on the pixel-level annotated **CRAFTS-FRT** dataset to precisely segment dispersive trajectories.
2.  **IMPIC Algorithm:** A physics-driven Iterative Mask-based Parameter Inference and Calibration algorithm that directly infers Dispersion Measure (DM) and Time of Arrival (ToA) from segmentation masks.



### 📂 Supported Data Formats
This library can process radio astronomy observation data in the following formats:
*   [Sigproc Filterbank](http://sigproc.sourceforge.net) (`.fil`)
*   [PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html) (`.fits`)

<!-- ## Key Features

*   **Unified Framework:** Seamlessly integrates signal detection and parameter estimation, eliminating the need for redundant DM-width grid searches.
*   **Physics-Driven Inference:** The **IMPIC** algorithm inverts geometric trajectory coordinates to infer physical parameters with high precision.
*   **High Sensitivity:** Achieves **98.0% recall** on the FAST-FREX benchmark, competitive with exhaustive search methods.
*   **False Positive Suppression:** Reduces false positives by over **99.9%** compared to PRESTO.
*   **Cross-Facility Generalization:** Successfully detected all 19 tested FRBs from **ASKAP** without retraining, demonstrating robustness across different telescopes.
*   **Efficiency:** Achieves processing speedups of up to **13.9×** compared to traditional pipelines. -->

## Installation

### Prerequisites
*   Python 3.10+
*   PyTorch 2.0+
*   CUDA 11.7+ 

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/BinZhang109/FRTSearch.git
cd FRTSearch

# Install MMDetection (Required for the detection module)
pip install -U openmim
mim install mmcv-full
pip install mmdet

# Install required packages
pip install -r requirements.txt

```
### Download Model Weights

⚠️ **First-time users must download the pre-trained model weights:**

The model is based on **Mask R-CNN** with **HRNet-W32** backbone. Download the weights from Hugging Face:

🔗 **[Download Model Weights from Hugging Face](https://huggingface.co/waterfall109/FRTSearch)**


Place the downloaded weights in the `models/` directory:

```
FRTSearch/
├── models/
│   └── mask_rcnn_hrnet_w32.pth
├── configs/
│   └── detector_FAST_19beams.py
└── ...
```


## Usage

Run the detection and inference pipeline using the following command:

```bash
python FRTSearch.py data.fits config.py --slide-size 128
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `data.fits` | Path to the observation data file (supports `.fits`, `.fil`) |
| `config.py` | Path to the model configuration file |
| `--slide-size` | (Optional) Number of subintegrations to read at once from the observation file (default: 128) |

## Datasets

We introduce **CRAFTS-FRT**, the first pixel-level annotated dataset dedicated to FRTs, derived from the Commensal Radio Astronomy FAST Survey.

| Property | Value |
|----------|-------|
| **Instances** | 2,392 (2,115 Pulsars, 15 RRATs, 262 FRBs) |
| **Source** | FAST 19-beam L-band receiver |
| **Availability** | [Download CRAFTS-FRT Dataset](https://doi.org/10.57760/sciencedb.Fastro.00038) |

<!-- ## Performance

Benchmarking on the **FAST-FREX** dataset (600 FRB bursts):

| Method | Recall (%) | FPPI (False Positives Per Image) | Speedup (vs PRESTO) |
|--------|------------|----------------------------------|---------------------|
| PRESTO | 79.8 | 4,983.5 | 1.0x |
| Heimdall | 85.0 | 49.0 | 13.8x |
| TransientX | 100.0 | 81.3 | 3.6x |
| **FRTSearch** | **98.0** | **4.1** | **25.5x** |

> **Note:** FRTSearch achieves a massive reduction in false positives while maintaining high recall, significantly alleviating the manual verification workload. -->

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

If you encounter any issues or have questions, please open an [Issue](https://github.com/BinZhang109/FRTSearch/issues).

## Citation

If you use FRTSearch or the CRAFTS-FRT dataset in your research, please cite our paper:

```bibtex
@article{zhang2025frtsearch,
  title={FRTSearch: Unified Detection and Parameter Inference of Fast Radio Transients using Instance Segmentation},
  author={Zhang, Bin and Wang, Yabiao and Xie, Xiaoyao ... and Wang, Pei and Li, Di},
  journal={Draft version (AASTeX631)},
  year={2025}
}
```

<div align="center">
  <sub>Exploring the dynamic universe with AI 🌌📡</sub>
</div>
