# 3D-Rotation-Equivariant Quaternion Neural Networks (PyTorch)

### Introduction
This repository is an official implementation of 3D-Rotation-Equivariant Quaternion Neural Networks
([arXiv](https://arxiv.org/abs/1911.09040),[SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_32#citeas)) which has been published at ECCV 2020.

Note that, we have found a bug and solved it. Therefore, the experimental results obtained based on this repository are slightly different from that in the paper. However, this did not essentially change our conclusions. New experimental results are as follows.

| Method     |                       | ModelNet40           |             |                       | 3D MNIST             |             |
| ---------- | --------------------- | -------------------- | ----------- | --------------------- | -------------------- | ----------- |
|            | Baseline w/o rotation | Baseline w/ rotation | REQNN(ours) | Baseline w/o rotation | Baseline w/ rotation | REQNN(ours) |
| PointNet++ | 25.87                 | 29.25                | **62.03**   | 44.19                 | 51.48                | **72.01**   |
| DGCNN      | 32.08                 | 33.78                | **84.57**   | 45.90                 | 50.00                | **84.57**   |
| PointConv  | 25.01                 | 26.46                | **81.93**   | 45.51                 | 48.08                | **85.71**   |

Besides, we have modified a few writing errors. In the second and third paragraph of the Experiment section, and the caption of Table 4, we have modified all “z-axis rotations” to “y-axis rotations”.

### Installation

<!--Install Pytorch. You may also need to install h5py. The code has been tested with Python 3.6, Pytorch 1.12.0, CUDA 10.0 and cuDNN 7.6 on Ubuntu 18.04.

Place datasets (ModelNet40 or 3D MNIST) to the `data` folder.-->

To run the program successfully, you need to include all packages in [requirements.txt](./requirements.txt) on your server.
```
pip install -r requirements.txt
```

### Usage

To train a model to classify point clouds.

Run the training script:


``` 1024 points
python main.py 
```

Log files and network parameters will be saved to `checkpoint` folder in default.


You can specify models, datasets and other train configurations in training script. For example:

`````` 
python main.py --exp_name=dgcnn_reqnn_1024_train --model=dgcnn_reqnn --dataset=modelnet --use_sgd=True --lr=0.001
``````

See HELP for the training script:

```
python main.py -h
```

Run the evaluation script with trained models:

``` 1024 points
python main.py --exp_name=dgcnn_reqnn_1024_eval --eval=True --eval_model_path=your_trained_model_path
```

### Citation

If you use this project in your research, please cite it.

```
@inproceedings{shen20193d,
	title={3d-rotation-equivariant quaternion neural networks},
	author={Shen, Wen and Zhang, Binbin and Huang, Shikun and Wei, Zhihua and Zhang, Quanshi},
	booktitle={ECCV},
	year={2020}
}
```
