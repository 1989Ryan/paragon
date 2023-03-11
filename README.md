# ParaGon: Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement

Project Page: [1989ryan.github.io/projects/paragon.html](https://1989ryan.github.io/projects/paragon.html)

This repository contains the pytorch implementation of the paper: Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement. 

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

You will need to have 269G free space to get all the data. 

```bash
python3 scripts/get_dataset.py
```

You can also choose to modify the script ``scripts/get_dataset.py`` to download testing data only (44G) if you do not have enough space. 

### Download pre-trained model

```bash
python3 scripts/pretrain_model.py
```

### Run the pre-trained model

```bash
bash scripts/run_pretrain.sh
```

### Training

```bash
bash scripts/train.sh
```

### Testing

```bash
bash scripts/eval.sh
```


## Citation

If you find this work useful in your research, please cite:

```
@InProceedings{zhao2023paragon,
    author    = {Zhao, Zirui and Lee, Wee Sun and Hsu, David},
    title     = {Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation},
    year      = {2023}
}
```

