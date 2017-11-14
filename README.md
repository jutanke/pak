![milk](https://user-images.githubusercontent.com/831215/32673460-9057f8ac-c64f-11e7-97e0-672eef1fe75d.png)

# pak
Personal computer vision/deep learning dataset helper toolbox to make it less tedious to download and 
load common datasets.

# Datasets

## MOT16

```python
from pak import datasets
mot16 = datasets.MOT16('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y_det, Y_gt  = mot16.get_train_raw("MOT16-02")
X, Y_det        = mot16.get_test_raw("MOT16-01")
```

![mot16](https://user-images.githubusercontent.com/831215/32783815-5336b2b4-c94d-11e7-8e8c-db4209e61450.png)

## 2DMOT15

```python
from pak import datasets
mot15 = datasets.MOT152D('/place/to/store/the/data')
# if the library cannot find the data in the given directory it
# will download it and place it there..

X, Y_det, Y_gt  = mot15.get_train_raw("ADL-Rundle-6")
X, Y_det        = mot15.get_test_raw("ADL-Rundle-1")
```

![mot15](https://user-images.githubusercontent.com/831215/32783818-5407e69a-c94d-11e7-9569-f6942b2be857.png)
