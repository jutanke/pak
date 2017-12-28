# Evaluation

This sub-framework helps with evaluating some of the results that
are based on the datasets in this library.

## MOTA

First described in [1].

```python
import numpy as np
from pak.evaluation import MOTA

# Data structure:
# [
#   (frame, pid, x, y),
#   ...
# ]
Gt = np.array([  # Ground-truth data
    [1, 1, 0.1, -0.1],
    [1, 2, 9.7, 10.1],
    [2, 1, 0.2, 0.3]
])
Hy = np.array([  # Hypothesis' 
    [1, 2, 10, 10],
    [1, 1, 0, 0],
    [2, 1, 0, 0]
])

threshold = 1

# result is a scalar value in the range of [-infinity, 1)
result = MOTA.evaluate(Gt, Hy, threshold)
```

# References
[1] Stiefelhagen, Rainer, et al. "The CLEAR 2006 evaluation." International Evaluation Workshop on Classification of Events, Activities and Relationships. Springer, Berlin, Heidelberg, 2006.
