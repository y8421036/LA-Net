# LA-Net: Layer Attention Network for 3D-to-2D Retinal Vessel Segmentation in OCTA Images


This work has been published at Physics in Medicine & Biology.

[DOI 10.1088/1361-6560/ad2011](https://iopscience.iop.org/article/10.1088/1361-6560/ad2011)

The implementation of this paper is based on PyTorch and verified under OCTA-500 dataset. 



|- root
|-- train.py - "The training code for LA-Net"
|-- test.py - "The test code for LA-Net"
|-- train+.py - "The training code for LA-Net+"
|-- test+.py - "The test code for LA-Net+"

First, train LA-Net using train.py, and the top 3 results will be saved in the logs/best_model directory. 
Then, use test.py to test and generate the features of the last layer of LA-Net. 
Next, train LA-Net+ using train+.py.
Finally, test LA-Net+ using test+.py.
