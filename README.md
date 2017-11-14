![milk](https://user-images.githubusercontent.com/831215/32673460-9057f8ac-c64f-11e7-97e0-672eef1fe75d.png)

# pak
Personal computer vision/deep learning dataset helper toolbox to make it less tedious to download and 
load common datasets.

## Datasets

### MOT16
Dataset[1] with 14 video sequences (7 train, 7 test) in unconstrained environments with both static and
moving cameras. Tracking + evaluation is done in image coordinates.

```python
from pak import datasets
mot16 = datasets.MOT16('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y_det, Y_gt  = mot16.get_train_raw("MOT16-02")
X, Y_det        = mot16.get_test_raw("MOT16-01")
```

![mot16](https://user-images.githubusercontent.com/831215/32783815-5336b2b4-c94d-11e7-8e8c-db4209e61450.png)

### 2DMOT2015

Dataset[2] with video sequences in unconstrained environments using static and moving cameras.

```python
from pak import datasets
mot15 = datasets.MOT152D('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y_det, Y_gt  = mot15.get_train_raw("ADL-Rundle-6")
X, Y_det        = mot15.get_test_raw("ADL-Rundle-1")
```

![mot15](https://user-images.githubusercontent.com/831215/32783818-5407e69a-c94d-11e7-9569-f6942b2be857.png)

### Market-1501
[3]Person re-identitification dataset collected in front of a supermarket from six different cameras. The dataset
contains 1501 different identities that are captured by at least two cameras.

```python
from pak import datasets
mot15 = datasets.Market1501('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y = m1501.get_train_raw()
```

![market1501](https://user-images.githubusercontent.com/831215/32785225-4afc5884-c951-11e7-95b1-542c11e7736e.png)


# References

[0] Milk-Icon: Icon made by Smashicons from www.flaticon.com

[1] Milan, Anton, et al. "MOT16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)

[2] Leal-Taixé, Laura, et al. "Motchallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015).

[3] Zheng, Liang, et al. "Scalable person re-identification: A benchmark." Proceedings of the IEEE International Conference on Computer Vision. 2015.
