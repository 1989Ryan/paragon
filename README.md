# ParaGon: Differentiable Parsing and Visual Grounding of Human Language Instructions for Object Placement

This repository contains the pytorch implementation of the paper: Differentiable Parsing and Visual Grounding of Human Language Instructions for Object Placement. 

## Quick start

### Docker

Build docker container
```bash
python scripts/docker_build.py
```
Run docker container
```bash
python scripts/docker_run.py
```

### Download dataset

```bash
bash scripts/data.sh
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
bash scripts/pretrain_model.sh
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

