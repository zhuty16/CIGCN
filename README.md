# Channel-Independent Graph Convolutional Network (CIGCN)

This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Leilei Sun, and Guoqing Chen. "Embedding Disentanglement in Graph Convolutional Networks for Recommendation." IEEE Transactions on Knowledge and Data Engineering (TKDE) (2021).

## Introduction
Channel-Independent Graph Convolutional Network (CIGCN) is a graph convolution-based recommendation framework that adopts diagonal filter matrices for learning disentangled user and item embeddings.

![](https://github.com/zhuty16/CIGCN/blob/master/framework.jpg)

## Citation
```
@article{zhu2021embedding,
  title={Embedding Disentanglement in Graph Convolutional Networks for Recommendation},
  author={Zhu, Tianyu and Sun, Leilei and Chen, Guoqing},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```

## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 1.5.0
* numpy == 1.14.2
* scipy == 1.1.0

## Dataset
* Download: [Google Drive](https://drive.google.com/drive/folders/1rcQOcl6K4q_n8584IDZ3qO5E6tjudEnG?usp=sharing)

## Example to Run the Codes
* Amazon Automotive dataset
```
python main.py --dataset=Automotive
```

