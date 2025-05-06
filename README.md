# Future Frame Prediction with SimVP

This project implements future frame prediction on traffic video data using the SimVP architecture. It builds upon the work presented in the CVPR'22 paper "SimVP: Simpler Yet Better Video Prediction" by Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z. Li.

## Overview

This repository contains an implementation of video prediction with modifications and adaptations for traffic video data. The core prediction model relies on the SimVP architecture from the [A4Bio/SimVP](https://github.com/A4Bio/SimVP) repository.

## Features

- Video frame aggregations
- Video frame prediction for traffic data
- Customized data processing for traffic video inputs
- Visualization of prediction results

## Dependencies

- PyTorch
- NumPy
- scikit-image
- tqdm
- Matplotlib (for visualization)

## Project Structure

- `API/`: Contains dataloaders and metrics
- `data/traffic/`: Traffic video data (not included in the repository due to size) should contain the video to train the model and the yolo model
- `model.py`: Implementation of the prediction model
- `modules.py`: Neural network modules
- `exp.py`: Core code for training, validation, and testing
- `main.py`: Main executable file with arguments

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/reetipd/FutureFramePrediction.git
cd FutureFramePrediction
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate SimVP
```

3. Prepare your traffic data in the `data/traffic/` directory (not included in the repository)

## Usage

To train the model:
```bash
python main.py --batch_size 16 --max_epoch 50 
```

## Credits

This project builds upon the SimVP architecture:

- **SimVP: Simpler Yet Better Video Prediction** - [GitHub Repository](https://github.com/A4Bio/SimVP)
- Paper: Gao, Z., Tan, C., Wu, L., & Li, S. Z. (2022). *SimVP: Simpler Yet Better Video Prediction*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3170-3180.

## Citation

If you use this code in your research, please cite the original SimVP paper:

```
@InProceedings{Gao_2022_CVPR,
    author    = {Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z.},
    title     = {SimVP: Simpler Yet Better Video Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3170-3180}
}
``` 