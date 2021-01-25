# Dual Self-Attention Network for Multivariate Time Series Forecasting

> **20.10.26 Update**: Due to the difficulty of installation and code maintenance caused by frequent updates of pytorch-lightning, **the code does not work correctly now**. We hope this code is useful for your reference, especially the part about the model, however, **we are sorry that we will no longer maintain the project**. We recommend you to refer to other similar applications of self-attention mechanism in time series, such as "[Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/abs/1907.00235)" and [https://github.com/maxjcohen/transformer](https://github.com/maxjcohen/transformer). 

This project is the PyTorch implementation of the paper "[DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting](https://dl.acm.org/citation.cfm?doid=3357384.3358132)", in which we propose a dual self-attention network (DSANet) for multivariate time series forecasting. The network architecture is illustrated in the following figure, and more details about the effect of each component can be found in the paper.

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/DSANet-model-structure.png)

## Requirements

* Python 3.5 or above
* PyTorch 1.1 or above
* pytorch-lightning

## How to run

You need to prepare the dataset first. Check [here](https://github.com/bighuang624/DSANet/blob/master/data/README.md).

```bash
# clone project
git clone https://github.com/bighuang624/DSANet.git

# install dependencies
cd DSANet
pip install requirements.txt

# run
python single_cpu_trainer.py --data_name {data_name} --n_multiv {n_multiv}
```

**Notice:** At present, we find that there are some bugs (presumably some problems left by the old version of pytorch-lightning) that make our code unable to run correctly on GPUs. You can currently run the code on the CPU as above.

## Citation

If our code is helpful for your research, please cite our paper:

```
@inproceedings{Huang2019DSANet,
  author = {Siteng Huang and Donglin Wang and Xuehan Wu and Ao Tang},
  title = {DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting},
  booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM 2019)},
  month = {November},
  year = {2019},
  address = {Beijing, China}
}
```

## Acknowledgement

Part of the code is heavily borrowed from [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
