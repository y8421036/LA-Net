# LA-Net: Layer Attention Network for 3D-to-2D Retinal Vessel Segmentation in OCTA Images

This work has been published at Physics in Medicine & Biology.

DOI： [10.1088/1361-6560/ad2011](https://iopscience.iop.org/article/10.1088/1361-6560/ad2011)

## Requirements
python=3.8.0

torch=2.0.1+cu117

## Data Preparation
* This paper is verified on the public OCTA-500 dataset.
* The OCTA-500 in the dataset download page provided in the paper has been updated by the data provider, that is, upgraded from two foregrounds (RV,FAZ) to four foregrounds (capillary,artery,vein,FAZ).
* To make it easier for you to reproduce LA-Net, I uploaded the image and label in previous dataset version to the network disk. In addition, you can also use the new version of OCTA-500. However, when using the new version, you should be aware that you cannot directly use the labeling part of the preprocessing code provided in this repository. You can determine whether your labeling processing is correct by taking the maximum projection of the OCTA volume data in the axial direction and checking whether the project image matches the new label.

### Dataset Download Link
OCTA-500(RV&FAZ): [https://pan.baidu.com/s/1nrfZt9zxmscL5ezgqhg5RQ?pwd=mavi](https://pan.baidu.com/s/1nrfZt9zxmscL5ezgqhg5RQ?pwd=mavi) 

Extraction code: mavi

### File tree of the dataset
datasets 
<br />└── OCTA-500 
<br />&emsp;    ├── 3M 
<br />&emsp;    │&emsp;   ├── OCT 
<br />&emsp;    │&emsp;   │&emsp;   ├── ... 
<br />&emsp;    │&emsp;   ├── OCTA 
<br />&emsp;    │&emsp;   │&emsp;   ├── ... 
<br />&emsp;    │&emsp;   ├── Label_RV 
<br />&emsp;    │&emsp;   │&emsp;   ├── ... 
<br />&emsp;    │&emsp;   └── GroundTruth 
<br />&emsp;    │&emsp;    &emsp;   ├── ... 
<br />&emsp;    └── 6M
<br />&emsp;     &emsp;   ├── OCT 
<br />&emsp;     &emsp;   │&emsp;   ├── ... 
<br />&emsp;     &emsp;   ├── OCTA 
<br />&emsp;     &emsp;   │&emsp;   ├── ...
<br />&emsp;     &emsp;   ├── Label_RV 
<br />&emsp;     &emsp;   │&emsp;   ├── ... 
<br />&emsp;     &emsp;   └── GroundTruth 
<br />&emsp;     &emsp;    &emsp;   ├── ... 

			
## Usage
* First, execute "preprocess.py" to rotate and label the files in GroundTruth folder. 
* Then, train LA-Net using "train.py", and the top 3 results will be saved in the logs/best_model directory.
* Next, use "test.py" to test and generate the features of the last layer of LA-Net. Note: the "test.py" file includes the npy parameter. When npy=1, it means generating features for training LA-Net+. When npy=2, it means generating the segmentation results of LA-Net.
* Then, train LA-Net+ using "train+.py".
* Finally, test LA-Net+ using "test+.py".


## Citation
If this code is helpful for your study, please cite:
```
@article{yang2024net,
  title={LA-Net: layer attention network for 3D-to-2D retinal vessel segmentation in OCTA images},
  author={Yang, Chaozhi and Li, Bei and Xiao, Qian and Bai, Yun and Li, Yachuan and Li, Zongmin and Li, Hongyi and Li, Hua},
  journal={Physics in Medicine \& Biology},
  volume={69},
  number={4},
  pages={045019},
  year={2024},
  publisher={IOP Publishing}
}
```
