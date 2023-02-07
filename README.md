# ParaGon: Differentiable Parsing and Visual Grounding of Human Language Instructions for Object Placement

This repository contains the pytorch implementation of the paper: Differentiable Parsing and Visual Grounding of Human Language Instructions for Object Placement. 

## Quick start

You are highly recommended to use Docker to run the code.  

### Docker

Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)

Build docker container
```bash
python3 scripts/docker_build.py
```
Run docker container
```bash
python3 scripts/docker_run.py
```

### Download dataset

```bash
python3 scripts/get_dataset.py
```

### Training

```bash
bash scripts/train.sh
```

### Testing

```bash
bash scripts/eval.sh
```
### Pre-trained models

```bash
python3 scripts/pretrain_model.py
```

## Citation

If you find this work useful in your research, please cite:

```
@InProceedings{zhao2023paragon,
    author    = {Zhao, Zirui and Lee, Wee Sun and Hsu, David},
    title     = {Differentiable Parsing and Visual Grounding of Human Language Instructions for Object Placement},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation},
    year      = {2023}
}
```

