---
layout: post
comments: true
title:  Decision Trees, Random Forest, Gradient Boosting
date:   2016-04-04 13:06:27
excerpt: What should I put here?
mathjax: true
---


## Decision Tree
Decision trees sound straightforward enough, as for example seen in Figure 1. It is however not that straightforward to implement. We try to do it here.  


<div class="imgcap">
<div>
<img src="/assets/tree/rpart.plot-example1.png">
</div>
<div class="thecap">Figure 1: An example of a decision tree - Credit: RPart </div>
</div>

## Codes
Assuming we are doing classification, \\(y \in {0,1} ^ n\\), and \\(X \in \mathbb{R} ^ {n \times p}\\)

A decision tree starts as follow:

```python
from __future__ import division
import numpy as np

def GetP(y):
    n = len(y)
    return (y.cumsum()[:-1] / np.arange(1,n))

def GetGini(y):
    n = len(y)
    p1 = GetP(y)
    p2 = GetP(y[::-1])[::-1]
    res = np.arange(1,n) * p1*(1 - p1) + np.arange(1,n)[::-1] * p2*(1 - p2)
    return res / n
```
