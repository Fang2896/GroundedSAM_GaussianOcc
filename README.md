## GroundedSAM for GaussianOcc

To generate per-pixel semantic labels from 2D image training data of **DDAD dataset** for our [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc), we provide the generation code with reference to [GroundedSAM_OccNeRF](https://github.com/JunchengYan/GroundedSAM_OccNeRF) and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

## Installation

You can also reference [GroundedSAM_OccNeRF](https://github.com/JunchengYan/GroundedSAM_OccNeRF) or [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to prepare the environment.

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Prepare environment:

```shell
git clone git@github.com:Fang2896/GroundedSAM_GaussianOcc.git
cd GroundedSAM_GaussianOcc/
pip install -r requirements.txt
```

Install Segment Anything:

```shell
python -m pip install -e segment_anything
```

Install GroundingDINO:

```shell
python -m pip install -e GroundingDINO
```

Other dependency

```shell
pip install diffusers transformers accelerate scipy safetensors
```

## Run the code

### Step1: Download the pretrained weights

Prepare weight for Segment Anything:

```shell
# Place it under GroundedSAM_GaussianOcc/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
```

### Step2: Prepare the models

You can download the models locally in advance or use the `huggingface api` to load the model

#### Load models from local

You can download the GroundiongDINO model and BERT model from [百度云盘(BaiduPan)](https://pan.baidu.com/s/1pzCAi_7SrDkPmU8ea31L1A?pwd=xkk3) or [Google Driver](). Then unzip the file under `GroundedSAM_GaussianOcc/`

```shell
unzip models--ShilongLiu--GroundingDINO.zip
unzip models--bert-base-uncased.zip
```

#### Load models from Hugging Face

You can also change the function `load_model_hf()` in `groundedsam_generate_sem_demo.py` and `groundedsam_generate_sem_ddad.py` to the original code in [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam.ipynb) to load GroundingDINO from Hugging Face. You should also change the function `get_pretrained_language_model()` in `GroundingDINO/util/get_tokenlizer.py` back to the code in [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/GroundingDINO/groundingdino/util/get_tokenlizer.py) to load BERT

### Step3: Generate semantic labels

#### Generate semantic labels on DDAD trainset

**Prepare data**

If you are using GroundedSAM_GaussianOcc **individually**, please link your DDAD dataset path to the `GroundedSAM_GaussianOcc/` folder and download metadata according to [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc), then modify the `data_path` in `groundedsam_generate_sem_ddad.py`.

```shell
ln -s DATA_PATH ./data
```

We use `groundedsam_generate_sem_ddad.py` to generate semantic labels of DDAD dataset for GaussianOcc self-supervised occupancy learning. Modify `sava_path` in `groundedsam_generate_sem_ddad.py` to determine where to save the results.
Running shell script is `run.sh`, you may need to modify the `config` parameter to your `ddad_volume.txt` location and make sure the python script is running under `GroundedSAM_GaussianOcc` folder (i.e. modify `cd your_folder_path/GaussianOcc` in `run.sh`)

```shell
bash run.sh 
```

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@article{gan2024gaussianocc,
  title={Gaussianocc: Fully self-supervised and efficient 3d occupancy estimation with gaussian splatting},
  author={Gan, Wanshui and Liu, Fang and Xu, Hongbin and Mo, Ningkai and Yokoya, Naoto},
  journal={arXiv preprint arXiv:2408.11447},
  year={2024}
}
```
