# Monodepth2

The reference PyTorch implementation for training and testing depth estimation models using the method described in:

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

```bibtex
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2019}
}
```

## Table of Contents

- [Setup](#setup)
- [Prediction for a Single Image](#prediction-for-a-single-image)
- [KITTI Training Data](#kitti-training-data)
- [Training](#training)
- [KITTI Evaluation](#kitti-evaluation)
- [Precomputed Results](#precomputed-results)
- [Visualization](#visualization)
- [License](#license)

## Setup

We recommend using `pyenv` and `pipenv` for managing Python versions and dependencies.

### Requirements

See `Pipfile.lock` for the full list of dependencies.

### Quick Install

```bash
# Install pyenv (via anyenv)
git clone https://github.com/anyenv/anyenv ~/.anyenv
echo 'export PATH="$HOME/.anyenv/bin:$PATH"' >> ~/.bash_profile
echo 'anyenv > /dev/null 2>&1 && eval "$(anyenv init -)"' >> ~/.bash_profile
anyenv install --init
anyenv install pyenv
pyenv install 3.9.1
pyenv global 3.9.1

# Install dependencies
pip install pipenv
cd /path/to/monodepth2
pipenv sync

# Test run
pipenv run python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

### GPU Compatibility

If you have a newer GPU, you may encounter the following error:

```
GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
```

In this case, install a compatible PyTorch version:

```bash
pipenv run pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Alternative: Anaconda Setup

If you prefer [Anaconda](https://www.anaconda.com/download/), install dependencies with:

```bash
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # needed for evaluation
```

Our experiments used PyTorch 0.4.1, CUDA 9.1, Python 3.6.6, and Ubuntu 18.04. The code is also compatible with PyTorch 1.0 and Python 2.7. For Python 3.7, we recommend creating a virtual environment with Python 3.6.6:

```bash
conda create -n monodepth2 python=3.6.6 anaconda
```

## Prediction for a Single Image

Run depth prediction on a single image:

```bash
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

On first run, the pretrained model (99MB) will be downloaded to the `models/` folder.

### Available Models

| Model Name | Training Modality | ImageNet Pretrained | Resolution | KITTI Abs. Rel. Error | delta < 1.25 |
|------------|-------------------|---------------------|------------|----------------------|--------------|
| [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip) | Mono | Yes | 640 x 192 | 0.115 | 0.877 |
| [`stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip) | Stereo | Yes | 640 x 192 | 0.109 | 0.864 |
| [`mono+stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip) | Mono + Stereo | Yes | 640 x 192 | 0.106 | 0.874 |
| [`mono_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip) | Mono | Yes | 1024 x 320 | 0.115 | 0.879 |
| [`stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip) | Stereo | Yes | 1024 x 320 | 0.107 | 0.874 |
| [`mono+stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip) | Mono + Stereo | Yes | 1024 x 320 | 0.106 | 0.876 |
| [`mono_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip) | Mono | No | 640 x 192 | 0.132 | 0.845 |
| [`stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip) | Stereo | No | 640 x 192 | 0.130 | 0.831 |
| [`mono+stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip) | Mono + Stereo | No | 640 x 192 | 0.127 | 0.836 |

Additional models trained on the odometry split: [monocular](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_odom_640x192.zip) and [mono+stereo](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_odom_640x192.zip).

ResNet-50 models: [ImageNet pretrained](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_640x192.zip) and [trained from scratch](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_no_pt_640x192.zip). Set `--num_layers 50` when using these.

## KITTI Training Data

Download the [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php):

```bash
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Unzip the files:

```bash
cd kitti_data
unzip "*.zip"
cd ..
```

**Note:** The dataset is approximately 175GB, so ensure you have sufficient disk space.

### Image Conversion

Our default settings expect JPEG images. Convert PNG to JPEG with:

```bash
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

**Important:** This command deletes the original PNG files. Alternatively, train from raw PNG files by adding the `--png` flag during training.

The explicit sampling factor `2x2,1x1,1x1` is specified because Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which produces different results.

You can store the KITTI dataset in any location and specify it with the `--data_path` flag.

### Splits

Train/test/validation splits are defined in the `splits/` folder. By default, the code uses [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split. You can also use the [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with the `--split` flag.

### Custom Dataset

Train on a custom dataset by creating a new dataloader class that inherits from `MonoDataset`. See `datasets/kitti_dataset.py` for reference.

## Training

By default, models and TensorBoard event files are saved to `~/tmp/<model_name>`. Change this with the `--log_dir` flag.

### Monocular Training

```bash
python train.py --model_name mono_model
```

### Stereo Training

For stereo-only training, use the full Eigen training set:

```bash
python train.py --model_name stereo_model \
  --frame_ids 0 --use_stereo --split eigen_full
```

### Monocular + Stereo Training

```bash
python train.py --model_name mono+stereo_model \
  --frame_ids 0 -1 1 --use_stereo
```

### GPU Configuration

The code runs on a single GPU. Specify which GPU to use:

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

All experiments were performed on a single NVIDIA Titan Xp.

| Training Modality | GPU Memory | Training Time |
|-------------------|------------|---------------|
| Mono | 9GB | 12 hours |
| Stereo | 6GB | 8 hours |
| Mono + Stereo | 11GB | 15 hours |

### Finetuning

Load an existing model for finetuning:

```bash
python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19
```

### Other Options

Run `python train.py -h` or see `options.py` for additional training options including learning rates and ablation settings.

## KITTI Evaluation

Prepare the ground truth depth maps:

```bash
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```

### Evaluating Models

For monocular models:

```bash
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono
```

For stereo models (see note below):

```bash
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_stereo
```

Results may vary slightly from published numbers due to randomization in weight initialization and data loading.

### Evaluation Splits

| Split | Test Set Size | For Models Trained With | Description |
|-------|---------------|-------------------------|-------------|
| `eigen` | 697 | `--split eigen_zhou` (default) or `--split eigen_full` | Standard Eigen test files |
| `eigen_benchmark` | 652 | `--split eigen_zhou` (default) or `--split eigen_full` | Improved ground truth from the [KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| `benchmark` | 500 | `--split benchmark` | [KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files |

For `--eval_split benchmark`, no scores are reported. Instead, PNG images are saved for upload to the evaluation server.

### External Disparities

Evaluate disparities from other methods:

```bash
python evaluate_depth.py --ext_disp_to_eval ~/other_method_disp.npy
```

### Note on Stereo Evaluation

Our stereo models use an effective baseline of `0.1` units, while the KITTI stereo rig has a baseline of `0.54m`. A scaling factor of `5.4` must be applied for evaluation. The `--eval_stereo` flag automatically disables median scaling and applies this scale factor.

### Odometry Evaluation

For models trained with `--split odom --dataset kitti_odom --data_path /path/to/kitti/odometry/dataset`:

Download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (color, 65GB) and ground truth poses. Ensure PNGs are converted to JPGs.

Evaluate with:

```bash
python evaluate_pose.py --eval_split odom_9 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
python evaluate_pose.py --eval_split odom_10 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
```

## Precomputed Results

Download precomputed disparity predictions:

| Training Modality | Input Size | File Size | Download |
|-------------------|------------|-----------|----------|
| Mono | 640 x 192 | 343 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192_eigen.npy) |
| Stereo | 640 x 192 | 343 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192_eigen.npy) |
| Mono + Stereo | 640 x 192 | 343 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192_eigen.npy) |
| Mono | 1024 x 320 | 914 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320_eigen.npy) |
| Stereo | 1024 x 320 | 914 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320_eigen.npy) |
| Mono + Stereo | 1024 x 320 | 914 MB | [Link](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320_eigen.npy) |

## Visualization

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir ~/tmp
```

Then open `localhost:6006` in your browser.

## License

Copyright 2019 Niantic, Inc. Patent Pending. All rights reserved.

Please see the [license file](LICENSE) for terms. This code is for non-commercial use only.
