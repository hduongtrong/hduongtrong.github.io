---
layout: post
comments: true
title:  Matrix Multiplication Benchmark
date:   2016-03-07 13:06:27
excerpt: OpenBlas, MKL, CPU, GPU
mathjax: true
---


## The setting

```python
import numpy as np
import time

n = 10000
x = np.random.randn(n,n)
a = time.time(); x.dot(x); print time.time() - a
```

The contestants

| CPU 					| Freq 		| Num-Cores | L3-Cache 	| Date | Price  | Passmark|
| ---------------------:| ---------:| ---------:| ---------:| ----:| ------:| -------:|
| Intel Core i5-4260U	| 1.40Ghz 	| 2 		| 3MB 		| Q2-14| 315  	| 3548	  |	 		