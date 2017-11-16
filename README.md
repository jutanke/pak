# pak
![milk](https://user-images.githubusercontent.com/831215/32673460-9057f8ac-c64f-11e7-97e0-672eef1fe75d.png)
Personal computer vision/deep learning dataset helper toolbox to make it less tedious to download and 
load common datasets.

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

X, Y_det, Y_gt  = mot16.get_train_raw("MOT16-02")
X, Y_det        = mot16.get_test_raw("MOT16-01")
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

X, Y_det, Y_gt  = mot15.get_train_raw("ADL-Rundle-6")
X, Y_det        = mot15.get_test_raw("ADL-Rundle-1")
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

X, Y = m1501.get_train_raw()
```

![market1501](https://user-images.githubusercontent.com/831215/32785225-4afc5884-c951-11e7-95b1-542c11e7736e.png)

### CUHK03
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

![pak_cuhk03](https://user-images.githubusercontent.com/831215/32894185-592e68e8-cadc-11e7-833b-d479ab5a9cd5.png)

#### Licence 
This dataset is ONLY released for academic use. Please do not further distribute the dataset (including the download link), or put any of the images on the public website, due to the university regulations and privacy policy in Hong Kong law. Please kindly cite our paper if you use our data in your research. Thanks and hope you will benefit from our dataset. 


# References

[0] Milk-Icon: Icon made by Smashicons from www.flaticon.com

[1] Milan, Anton, et al. "MOT16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)

[2] Leal-Taixé, Laura, et al. "Motchallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015).

[3] Zheng, Liang, et al. "Scalable person re-identification: A benchmark." Proceedings of the IEEE International Conference on Computer Vision. 2015.
