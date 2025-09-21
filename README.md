# [TOMM 2025] MambaVesselNet++: A Hybrid CNN-Mamba Architecture for Medical Image Segmentation

[[`arXiv`](https://arxiv.org/abs/2507.19931)] [[`ACM Transactions`](https://dl.acm.org/doi/abs/10.1145/3757324)] 

![process.png](imgs/archi.png)

## 📚Data Preparation
This study uses the publicly available datasets related to cerebrovascular segmentation. Accessible at [https://xzbai.buaa.edu.cn/datasets.html](https://xzbai.buaa.edu.cn/datasets.html). 

## 🎪Quickstart

### Install casual-conv1d

please make sure the cuda version ≥ 11.6

```bash
cd casual-conv1d
python setup.py install
```

### Install Mamba

```bash
cd mamba
python setup.py install
```

### Install MONAI

```bash
pip install monai
```

## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```
@article{xu2025mambavesselnet++,
  title={MambaVesselNet++: A Hybrid CNN-Mamba Architecture for Medical Image Segmentation},
  author={Xu, Qing and Chen, Yanming and Li, Yue and Liu, Ziyu and Lou, Zhenye and Zhang, Yixuan and Zheng, Huizhong and He, Xiangjian},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2025},
  publisher={ACM New York, NY}
}
```
If you have any questions about our project, feel free to contact me by email at [qing.xu@nottingham.edu.cn](mailto:qing.xu@nottingham.edu.cn) or [yanming.chen@yale.edu](mailto:yanming.chen@yale.edu) 

## Acknowledgements

* [Mamba](https://github.com/state-spaces/mamba)


