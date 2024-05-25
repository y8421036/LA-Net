# LA-Net: Layer Attention Network for 3D-to-2D Retinal Vessel Segmentation in OCTA Images


This work has been published at Physics in Medicine & Biology.

[DOI 10.1088/1361-6560/ad2011](https://iopscience.iop.org/article/10.1088/1361-6560/ad2011)

The implementation of this paper is based on PyTorch and verified under OCTA-500 dataset. 


 <br />
|- root <br />
|-- train.py - "The training code for LA-Net." <br />
|-- test.py - "The test code for LA-Net." <br />
|-- train+.py - "The training code for LA-Net+." <br />
|-- test+.py - "The test code for LA-Net+." <br />

First, train LA-Net using train.py, and the top 3 results will be saved in the logs/best_model directory.  <br />
Then, use test.py to test and generate the features of the last layer of LA-Net.  <br />
\qquad Note: the test.py file includes the npy parameter. When npy=1, it means generating features for training LA-Net+. When npy=2, it means generating the segmentation results for LA-Net.  <br />
Next, train LA-Net+ using train+.py. <br />
Finally, test LA-Net+ using test+.py. <br />
