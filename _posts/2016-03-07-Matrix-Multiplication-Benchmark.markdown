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

| CPU 					| Freq 		| N\_Cores  | L3\_Cache | Date | Price  | Passmark|
| ---------------------:| ---------:| ---------:| ---------:| ----:| ------:| -------:|
| Intel Core i5\-4260U	| 1.4GHz 	| 2 		| 3MB 		| Q2-14| 315    | 3548	|
| Intel Xeon E5\-2643 v2| 3.5GHz    | 2x6=12    | 2x25MB 	| Q3-13| 2x1552 | 2x11735 |
| AMD Opteron 8384 		| 2.7GHz 	| 8x4=32 	| 8x6MB 	| Q4-08| 8x2149 | NA 		|
| AMD Opteron 8272 		| 1.4Ghz 	| 2x8=16 	| 2x6MB 	| Q4-11| 2x523  | 2x6748  |

## MKL vs OpenBlas

Here are the running time in seconds. The number in () are roughly the fluctuation of running time. For the GPU result, Tesla K80 is a dual GPU, and this is only using one of them, which is equivalent to Tasla K40. In addition, calculation is carried out with float64, which GPU is bad at. Note for non-mkl, we use the default Blas library on OS X El Capitan. For the other, they are Openblas. MKL in general gives more variable results, but slightly better than the non-MKL on Intel CPUs. 

| CPU 					| Non-MKL 	| MKL  		| 
| ---------------------:| ---------:| ---------:| 
| Intel Core i5\-4260U	| 43 	 	| 32 		| 
| Intel Xeon E5\-2643 v2| 15.6     	| 10.4 (3) 	| 
| AMD Opteron 8384 		| 15.4 (2)	| 12.3 (1)	| 
| AMD Opteron 8272 		| 17.3		| 22 (5) 	|
| Tesla K-80 			| 16.3 		| 			| 

## CPU vs GPU

To really see the power of GPU, we use float32 instead.

| Matrix dim 			| CPU 		| GPU Tensorflow | GPU Skcuda | 
| ---------------------:| ---------:| --------------:| ----------:|
| 10000					| 6.3 	 	| 2.3 			 | 1.3		  |
| 15000			 		| 17		| 6.8 			 | 3.7 		  |
| 20000					| 39     	| 10.8 			 | 8.32 	  |
| 30000					| 122     	| NA 			 | 27.0 	  |

GPU only provides a speed up of around 4-5 times. The GPU 1 is done by Tensorflow, which might not be very efficient. The GPU 2 is done by Scikit-cuda, which is a wrapper for pycuda. For the later one, we also see a breakdown of communication time between CPU and GPU. It spends around 15% of the time copying data in and out of GPU. 

Tools for doing linear algebra on GPU.  

1. Pycuda: this is the lowest level, a wrapper of CUDA for Python
2. Scikit-cuda: a wrapper over pycuda
3. Cula: provide LAPACK type of matrix multiplication for CUDA
4. Numbapro / accelerate: from Anaconda
5. Theano / Tensorflow

## Codes
#### Skcuda + Pycuda

```python
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np, time
import skcuda.linalg as culinalg
import skcuda
culinalg.init()

dim = 30000
rnd = np.random.RandomState(0)
a = rnd.rand(dim, dim).astype(np.float32)

start = time.time()
a_gpu = gpuarray.to_gpu(a)
print 'Copy in', time.time() - start

start = time.time()
rescpu = np.dot(a, b)
print 'CPU:', time.time() - start

start = time.time()
resgpu = culinalg.dot(a_gpu, a_gpu)
print skcuda.misc.sum(resgpu)
print 'GPU:', time.time() - start

start = time.time()
resgpu = resgpu.get()
print 'Copy out', time.time() - start
print np.allclose(rescpu, resgpu)
print np.allclose(resgpu, rescpu)
```

#### Tensorflow
```python
import numpy as np
import tensorflow as tf
import time

X = tf.constant(np.array(np.random.randn(20000,20000), dtype = np.float32), dtype = tf.float32)
Y = tf.matmul(X, tf.transpose(X))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

a = time.time()
sess.run(Y)
print time.time() - a
```

#### CPU (both OpenBLAS and MKL)
```python
import numpy as np
import time
np.random.seed(1)
n = 30000
x = np.array(np.random.randn(n,n), dtype = np.float32)
a = time.time(); x.dot(x); print time.time() - a
```