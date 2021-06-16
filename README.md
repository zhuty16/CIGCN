# Channel-Independent Graph Convolutional Network (CIGCN)

This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Leilei Sun, and Guoqing Chen. "Embedding Disentanglement in Graph Convolutional Networks for Recommendation." IEEE Transactions on Knowledge and Data Engineering (TKDE) (2021).

## Introduction
Channel-Independent Graph Convolutional Network (CIGCN) is a graph convolution-based recommendation framework that adopts diagonal filter matrices for learning disentangled user and item embeddings.

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

## Example to Run the Codes
* Amazon Automotive dataset
```
python main.py --dataset=Automotive
```

