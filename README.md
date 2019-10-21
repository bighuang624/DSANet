# Dual Self-Attention Network for Multivariate Time Series Forecasting

**We are reorganizing the code. The code will be released soon.**

This project is the code of the paper "[DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting]()", in which we propose a dual self-attention network (DSANet) for multivariate time series forecasting. The network architecture is illustrated in the following figure, and more details about the effect of each component can be found in the paper.

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/DSANet-model-structure.png)

## Requirements

* Python 3.5 or above
* PyTorch

## Results

### Evaluation Results

With *window* = 32:

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/exp_results_window_32.png)

With *window* = 64:

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/exp_results_window_64.png)

With *window* = 128:

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/exp_results_window_128.png)

### Ablation Study

With *window* = 32:

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/ablation_RRSE.png)

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/ablation_MAE.png)

![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/ablation_CORR.png)

## Citation

If our codes are helpful for your research, please cite our paper:

```
@inproceedings{Huang2019DSANet,
  author = {Huang, Siteng and Wang, Donglin and Wu, Xuehan and Tang, Ao},
  title = {DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting},
  booktitle = {The 28th ACM International Conference on Information and Knowledge Management (CIKM ’19)},
  month = {November},
  year = {2019},
  address = {Beijing, China}
}
```

ACM Reference Format: 

Siteng Huang, Donglin Wang, Xuehan Wu, and Ao Tang. 2019. DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting. In The 28th ACM International Conference on Information and Knowledge Management (CIKM ’19), November 3–7, 2019, Beijing, China. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3357384.3358132