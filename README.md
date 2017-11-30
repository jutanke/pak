# pak
![milk](https://user-images.githubusercontent.com/831215/32673460-9057f8ac-c64f-11e7-97e0-672eef1fe75d.png)
Personal computer vision/deep learning dataset helper toolbox to make it less tedious to download and 
load common datasets. This software is not affiliated with any of the datasets but is instead just a thin helper box to ease 
interacting with the data. Please respect the respective dataset author's licenses!

## Install
Install the library using pip:
```bash
pip install git+https://github.com/justayak/pak.git
```

### Requirements 

* python >=3.5
* numpy
* scipy
* skimage
* h5py


## Datasets

### MOT16
[Dataset](https://motchallenge.net/)[1] with 14 video sequences (7 train, 7 test) in unconstrained environments with both static and moving cameras.
Tracking + evaluation is done in image coordinates.
[Sample code](https://github.com/justayak/pak/blob/master/samples/MOT16.ipynb)

```python
from pak import datasets
mot16 = datasets.MOT16('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

# Some videos might be too large to fit into your memory so you
# can load them into a memory-mapped file for easier handling
X, Y_det, Y_gt  = mot16.get_train("MOT16-02", memmapped=True)
X, Y_det        = mot16.get_test("MOT16-01", memmapped=True)
```

![mot16](https://user-images.githubusercontent.com/831215/32783815-5336b2b4-c94d-11e7-8e8c-db4209e61450.png)

#### License

[Creative Commons Attribution-NonCommercial-ShareAlike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). 
This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options.

### 2DMOT2015

[Dataset](https://motchallenge.net/)[2] with video sequences in unconstrained environments using static and moving cameras.
[Sample code](https://github.com/justayak/pak/blob/master/samples/MOT15_2D.ipynb)

```python
from pak import datasets
mot15 = datasets.MOT152D('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y_det, Y_gt  = mot15.get_train("ADL-Rundle-6")
X, Y_det        = mot15.get_test("ADL-Rundle-1")
```

![mot15](https://user-images.githubusercontent.com/831215/32783818-5407e69a-c94d-11e7-9569-f6942b2be857.png)

#### License

[Creative Commons Attribution-NonCommercial-ShareAlike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). 
This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options.

### Market-1501
[3]Person re-identitification dataset collected in front of a supermarket from six different cameras. The dataset
contains 1501 different identities that are captured by at least two cameras.
[Sample code](https://github.com/justayak/pak/blob/master/samples/Market1501.ipynb)

```python
from pak import datasets
mot15 = datasets.Market1501('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y = m1501.get_train()
```

![market1501](https://user-images.githubusercontent.com/831215/32785225-4afc5884-c951-11e7-95b1-542c11e7736e.png)

### CUHK03
Dataset[4] with cropped images of persons from different angles and cameras. The dataset uses human annotated as well as
automatically annotated pictures.
The authors require users to explicitly download the data from their [website](https://docs.google.com/forms/d/e/1FAIpQLSfueNRWgRp3Hui2HdnqHGbpdLUgSn-W8QxpZF0flcjNnvLZ1w/viewform?formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0) so it is not possible to automatically download it.
[Sample code](https://github.com/justayak/pak/blob/master/samples/CUHK03.ipynb)

```python
from pak import datasets

# the images do not have the same size and have to be resized
w, h = 60, 160
cuhk03 = datasets.cuhk03('/place/where/the/downloaded/zip/is/stored', target_w=w, target_h=h)

X, Y = cuhk03.get_labeled()
# X, Y = cuhk03.get_detected()
```

#### Licence 
This dataset is ONLY released for academic use. Please do not further distribute the dataset (including the download link), or put any of the images on the public website, due to the university regulations and privacy policy in Hong Kong law. Please kindly cite our paper if you use our data in your research. Thanks and hope you will benefit from our dataset. 


### Leeds Sports Pose Extended Training Dataset
[5][Dataset](http://sam.johnson.io/research/lspet.html)

[Sample code](https://github.com/justayak/pak/blob/master/samples/LeedsSportsPoseExtended.ipynb)

```python
from pak import datasets
lspe = datasets.LSPE('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y = lspe.get_raw()
```

![pak_lspe](https://user-images.githubusercontent.com/831215/32917435-77d235ac-cb1f-11e7-957e-8fcd8301e271.png)

### DukeMTMC-reID
[6][Dataset](https://drive.google.com/uc?id=0B0VOCNYh8HeRdnBPa2ZWaVBYSVk&export=download) that is similar to market-1501.
However, the images are not cropped to have the same size!

[Sample code](https://github.com/justayak/pak/blob/master/samples/DukeMTMC-reID.ipynb)

```python
from pak import datasets
duke = datasets.DukeMTMC_reID('/place/to/store/the/data')
# please download the dataset from the given url and put the zip file into the path

X, Y = duke.get_test()
```

![duke](https://user-images.githubusercontent.com/831215/33133206-5fd89e88-cf9c-11e7-930c-65ef51ae061a.png)

#### License

ATTRIBUTION PROTOCOL

If you use the DukeMTMC-reID data or any data derived from it, please cite the original work as follows:

```
@article{zheng2017unlabeled,
  title={Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  journal={arXiv preprint arXiv:1701.07717},
  year={2017}
}
```

and include this license and attribution protocol within any derivative work.


If you use the DukeMTMC data or any data derived from it, please cite the original work as follows:

```
@inproceedings{ristani2016MTMC,
  title =        {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author =       {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle =    {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year =         {2016}
}
```

and include this license and attribution protocol within any derivative work.


The DukeMTMC-reID dataset is derived from DukeMTMC dataset (http://vision.cs.duke.edu/DukeMTMC/).

The DukeMTMC-reID dataset is made available under the Open Data Commons Attribution License (https://opendatacommons.org/licenses/by/1.0/) and for academic use only.

READABLE SUMMARY OF Open Data Commons Attribution License 

You are free:

    To Share: To copy, distribute and use the database.
    To Create: To produce works from the database.
    To Adapt: To modify, transform and build upon the database.

As long as you:

    Attribute: You must attribute any public use of the database, or works produced from the database, in the manner specified in the license. For any use or redistribution of the database, or works produced from it, you must make clear to others the license of the database and keep intact any notices on the original database.

### Hand Dataset
[7][Collection](http://www.robots.ox.ac.uk/~vgg/data/hands/) of hand images from various public images. 
[Sample code](https://github.com/justayak/pak/blob/master/samples/hand_dataset.ipynb)

```python
from pak import datasets
hand = datasets.Hand('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X_test, Y_test = hand.get_test()
X_train, Y_train = hand.get_train()
X_val, Y_val = hand.get_val()
```

![hand_dataset](https://user-images.githubusercontent.com/831215/33312758-13f0f7f4-d429-11e7-9561-3cebf8832548.png)


### EgoHands Dataset
[8][Dataset](http://vision.soic.indiana.edu/projects/egohands/) containing 48 Google Glass videos of interactions
between two people.
[Sample code](https://github.com/justayak/pak/blob/master/samples/EgoHands.ipynb)

```python
from pak import datasets
egohand = datasets.EgoHands(root)
# if the library cannot find the data in the given directory it
# will download it and place it there..

# the image data occupies ~15GB of RAM so if your computer
# cannot handle such big data you should set memmapped=True
X, Y = egohand.get_raw(memmapped=True)
```

![egohands](https://user-images.githubusercontent.com/831215/33382371-da8a84a0-d520-11e7-8e87-95c5aba8e814.png)


# References

[0] Milk-Icon: Icon made by Smashicons from www.flaticon.com

[1] Milan, Anton, et al. "MOT16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)

[2] Leal-Taixé, Laura, et al. "Motchallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015).

[3] Zheng, Liang, et al. "Scalable person re-identification: A benchmark." Proceedings of the IEEE International Conference on Computer Vision. 2015.

[4] Li, Wei, et al. "Deepreid: Deep filter pairing neural network for person re-identification." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014.

[5] Johnson, Sam, and Mark Everingham. "Learning effective human pose estimation from inaccurate annotation." Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.

[6] Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European Conference on Computer Vision. Springer International Publishing, 2016.

[7] Mittal, Arpit, Andrew Zisserman, and Philip HS Torr. "Hand detection using multiple proposals." BMVC. 2011.

[8] Bambach, Sven, et al. "Lending a hand: Detecting hands and recognizing activities in complex egocentric interactions." Proceedings of the IEEE International Conference on Computer Vision. 2015.
