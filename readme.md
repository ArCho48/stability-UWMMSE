# UWMMSE-stability
Tensorflow implementation of Stability Analysis of UWMMSE

## Overview
This library contains a Tensorflow implementation of the paper Stability Analysis of Unfolded WMMSE for Power Allocation[1].
## Dependencies

* **python>=3.6**
* **tensorflow>=1.14.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [main](https://github.com/ArCho48/stability-UWMMSE/blob/main/main.py): Main code for generating dataset and training/evaluating UWMMSE model. Run as *python3 main.py* \[dataset ID\] \[exp ID\] \[mode\]. Eg., to train UWMMSE on dataset with ID set3, run *python3 main.py set3 uwmmse train*. Generates dataset with given ID if not already present.
* [validate](https://github.com/ArCho48/stability-UWMMSE/blob/main/validate.py): Plot figures 1(a) \& 1(b) in the paper. Run as *python3 main.py* \[dataset ID\]. Eg., to run on dataset with ID set3, run *python3 validate.py set3*
* [model](https://github.com/ArCho48/stability-UWMMSE/blob/main/model.py): Defines the UWMMSE model.
* [data](https://github.com/ArCho48/stability-UWMMSE/blob/main/data): should contain your dataset in folder {dataset ID}. 
* [models](https://github.com/ArCho48/stability-UWMMSE/blob/main/models): Stores trained models in a folder with same name as {datset ID}.
* [results](https://github.com/ArCho48/stability-UWMMSE/blob/main/results): Stores results in a folder with same name as {datset ID}.

## Usage


Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:arindam.chowdhury@rice.edu).

## Citation
```
[1] Chowdhury A, Gama F, Segarra S. Stability Analysis of Unfolded WMMSE for Power Allocation. 
arXiv preprint arXiv:2110.07471 2021 Oct 14.
```

BibTeX format:
```
@article{chowdhury2021stability,
  title={Stability Analysis of Unfolded WMMSE for Power Allocation},
  author={Chowdhury, Arindam and Gama, Fernando and Segarra, Santiago},
  journal={arXiv e-prints},
  pages={arXiv--2110},
  year={2021}
}


```
