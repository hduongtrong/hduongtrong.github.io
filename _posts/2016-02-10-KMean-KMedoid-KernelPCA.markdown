---
layout: post
comments: true
title:  PCA, Kernel PCA, MDS, K-Means, K-Medoids, Hierarchical Clustering
date:   2016-02-10 18:44:27
excerpt: Summary of Spectral Clustering
mathjax: true
---

## 1. PCA


<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/fig_pca_principal_component_analysis.png">
</div>
<div class="thecap">PCA illustration. Image Credit: Google Image </div>
</div>


### 1.1. The algorithm
Given data \\( X \in \mathbb{R} ^ {n \times p} \\), PCA process is as followed (the SVD version)

1. Center and rescale \\( X \\) to have zero mean, standard deviation one
2. Perform SVD \\( X = UDV ^ T \\), where \\( U \in \mathbb{R} ^ {n \times p}\\), \\( D\\) is a diagonal matrix of size \\( (p \times p ) \\), and V is a orthogonal matrix size \\( p \times p \\)
3. The projected data is \\( X \_ {pca} = UD \\) for as many column of U as the dimension we want to project 

U is called the loadings. UD is called the scores (projected X). U is also called the principle components. It is the eigenvectors of the covariance matrix. 

An analogous version of PCA (the eigen-decomposition version)

1. Center and rescale \\( X \\)
2. Construct covariance matrix \\( C = \frac{1}{n} X ^ T X \\)
3. Perform eigen-decomposition \\( C = V \Lambda V ^ T \\)
4. The projected data is \\( X \_ {pca} = X U \\) for as many column of U as the dimension we want to project 

To see while they are equivalent, note that the \\( U \\) from SVD and eigen-decomposition are the same, and that \\( \Lambda = D ^ 2\\). A such \\( X\_ {pca} = XU = UDV V ^ T = UD \\). 

### 1.2. Interpretation of PCA

* The first component \\( w ^ T X \\) is the dimension that maximize the variance, i.e.
$$\max \_ {\\| w \\| = 1} Var (w ^ T X ) = \max \_  {\\| w \\| = 1} \frac{1}{n} w ^ T X ^ T X w .$$
And the above expression is maximized when w is the largest eigenvector of \\( X ^ T X\\). At that value, the expression is equal to the eigenvalue of the covariance matrix. 
* The PCA projection is a set of points in lower dimension that best preserve the pair-wise distances. 
* PCA can also be interpreted in the sense compressing data, i.e. we post-multiplying matrix \\(X \\) with some vector \\( U \\)) to lower dimension for compression, then to obtain an approximate of the original data, we multiply this lower dimension projection with \\( U ^ {-1}\\). Then out of all the linear projection, PCA is the one that best preserve the original dataset when going back to the original space from the compressed data. For example images data. 

### 1.3. Properties
* It is fast. In the case of tall matrix, \\( p < n\\), the algorithm is \\( O(np ^ 2)\\). It scales well with the number of dimention
* The prefer way to do PCA is through SVD
* There is efficient library (ARPACK) to get the first few eigenvalues and eigenvectors. 
* Just like Linear Regression, one can also do Sparse PCA, Kernel PCA, Ridge PCA. 
* It is often used to reduce the dimension then run some clustering algorithm on the data 

## 2. Kernel PCA 

<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/KernelPCA.png">
</div>
<div class="thecap">Kernel PCA illustration. Image Credit: Google Image </div>
</div>
### 1.1. The Algorithm

1. Given a Kernel, i.e. the Radial Basis kernel 
$$ K (x, y) = \exp \left( - \frac {\\| x - y\\| ^ 2 }{\sigma ^ 2 } \right) $$, we construct the \\( n \times n \\) Kernel matrix \\( K\\), where \\( K\_{ij} = K(x\_i, x\_j)\\)
2. Double centering matrix \\( K\\) to have column mean and row mean zero 
$$ \tilde{K}  = K - 1 _ n ^ T K - K 1 _ n + 1 _ n ^ T  K 1 _ n$$, for \\( 1 _ N \\) is the vector of all 1. 
3. Solve for eigenvector and eigenvalues of \\( \tilde{K} / n\\)
$$ \frac{1}{N} \tilde{K} a\_k = \lambda \_ k a\_ k $$
4. The projected dataset is \\( X \_ {kpca} = \sum _ {i = 1} ^ {n} a \_ {ki} K(x, x\_i)\\)

### 1.2. Interpretation 

1. Linear Kernel PCA is equivalent to PCA as expected
2. Similar to other Kernel method, Kernel PCA can be thought of equivalent to doing PCA on a higher dimension feature space, with the same number of parameters (\\(p\\)). For example if we use the quadratic polynomial, the kernel function is just \\( K(x, y) = (x ^ T y + c) ^ 2\\)it is similar to feature engineer the dataset into \\( x\_1 ^ 2, x\_2 ^ 2, x\_1 x\_2, x\_1, x\_2 \\\), then do PCA up here with the constraint on the weight (so the weight on this new space can't be freely chosen, but only have 2 degree of freedom, i.e. the weight for the quadratic term must be the square the weight of the linear term). So it effectively performs PCA on a subspace of a higher dimension space, where the rank of the subspace is equal to \\( p \\).

### 1.3. Propreties 
1. Its memory requirement is quadratic in \\(n \\), that is expensive. The computation complexity is also (at least) \\( O (n ^ 2)\\). 
2. It is more flexible than linear PCA in the sence non-linear model is more flexible than linear regression

## 3. Multidimensional Scaling

<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/mds.gif">
</div>
<div class="thecap">From distance matrix of cities, reconstruct their location with MDS. Image Credit: Google Image </div>
</div>


### 3.1. Overview
It is very similar to PCA, or specifically Kernel PCA, so it is worth noting the fundamental difference. In Kernel PCA, one start with a data matrix \\( X \\), one then construct the kernel matrix, \\( XX ^ T\\) in case of linear kernel, then get get the projected dataset from the eigen-decomposition of this dataset. 
In MDS, the original dataset is unknown, we only know the distance matrix, now one wish to get a projected dataset that also best preserve the original unknown dataset, in the sense that the pairwise distances of the new dataset match the known distance matrix. 

One example when this might arise is in socialogy. We have ancient towns in England, where now we don't know exactly their location. But we have some measure of pairwise distance between two towns, based on how many married couples are between these two towns. From this measure of similarity, one wish to reconstruct the original location of the town. Of course this can only be done up to a translation and rotation (since doing these does not change the distances). 

### 3.2. The algorithm
From the distance matrix \\( D \in \mathbb{R ^ {n \times p}}\\) of some unknown data \\( Z \\) living in unknown dimension space \\( \mathbb{R} ^ p\\), one wish to construct \\( X \\) that takes \\( D \\) as its distance matrix. If the unknown data \\( Z \\) is of rank p, then a theorem state that we can get \\( X \\) uniquely up to a rotation and translation in \\( \mathbb{R} ^ p \\) as well. In higher dimension, of course there are infinitely many solution. In lower dimension, we can't get the exact distance matrix, but only wish to get data that best preserve the distance matrix. We can assume that \\( X,Z \\) have column mean zero. 

1. Double-centering \\( D ^ 2\\) to obtain \\( S = -\frac{1}{2} H D ^ 2 H\\),  for \\(H = I _ n - \frac{1}{n} 11 ^ T\\). Theorem 1: \\( S = XX ^ T = ZZ ^ T\\)
2. Diagonalize \\( S = U \Lambda U ^ T\\)
3. Using the first \\(p\\) eigenvalues, eigenvectors (in decreasing order), and obtain \\( X = U \Lambda ^ {1/2}\\). We can already see handwavingly that \\( XX ^ T = U \Lambda U ^ T = S\\) if we construct \\( X \\) this way. Theorem: X takes D as its distance matrix.
4. (Optional) If we only use the first \\( k\\) eigenvalue, eigenvector pair to construct \\(X\\), then Theorem: this X best preserves the original distance matrix \\(D\\), in the sense of minimizing sum of square of error. 

### 3.3. Properties
1. It is \\( O(n ^ 2)\\). 
2. If D is obtained from some known dataset \\( Z \\), then doing the above MDS is exactly equivalent to linear Kernel PCA
3. In practice, one construct the distance matrix using some non-Euclidean distance, then do MDS on this distance matrix. 
4. To emphasize close distance, we can do the same trick as in Spectral Clustering (see  Spectral Clustering post), transforming the distance to similarity matrix such that closer distance stay roughly as 1, while faraway distance are effectively 0. 

## 4. K-Means
<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/kmeans.png">
</div>
<div class="thecap">K-Means obtains convex cluster. Image Credit: Google Image </div>
</div>


### 4.1. Algorithm
Objective 
$$\min _ {z,u} J(z, u) = \sum _ {i = 1} ^ n \sum _ {j = 1} ^ k z _ i \\| x _ i - \mu _ j\\| ^ 2$$

This algorithm will result in a local solution to the above objective. The cost to obtain the global solution is NP. 

1. Given \\(k\\) the number of cluster, pick \\( k\\) points randomly from the dataset to be the center
2. Update \\( z _ i\\): associate each of n points with the closest center
3. Update \\( \mu _ j\\): recalculate the center as the mean of the points in that cluster
4. Repeat 2 and 3 until convergence
5. (Optional) Do the above procedues for different initial starting points and pick the best configuration in term of the objective 

### 4.2. Interpretation 
1. K-Means basically minimize the sum of square Euclidean distance from each point in the cluster to the center. The cluster will always be convex. 
2. It works well when the cluster are convex and somewhat separable 
3. The objective is minimized in a alternative minimization manner: we fix one coordinate, optimize the function along the other coordinate, then we fix the other coordinate and optimize along this coordinate. This approach is used in many non-convex optimization problem, and is very similar to EM algorithm. 

### 4.3. Properties
1. Its complexity is O(nkip) for i is the number of iteration we run. So it is fast. 
2. K-Mean kinda only work for Euclidean distance, since we take advantage of the cheap "finding center" part. For Euclidean distance, the new center that minimizes the sum of square distance between the center and its members is just the mean of all members. It is hard for other distance metrics. 
3. It suffer from outlier as Linear Regression suffers from outlier, or PCA suffer from outliers. They are all minimizing the sum of square errors. 

## 5. K-Medoids

<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/kmedoids.gif">
</div>
<div class="thecap">Difference between K-Medoids and K-Means. Image Credit: Google Image </div>
</div>
### 5.1. Overview
K-medoids at first sight look like K-Means with a different distance metrics. It is not quite that. The first difference is that where the center of K-Means can be any point in the space, the center of K-medoid must be one in the dataset. Secondly, K-medoid will works only with the distance matrix, K-means will need the exact location of each point to calculate the distance from that point to any arbitrary center. 

### 5.2. The Algorithm,

1. Given \\(k\\) the number of cluster, pick \\( k\\) points randomly from the dataset to be the center
2. Update \\( z _ i\\): associate each of n points with the closest center
3. Update \\( \mu _ j\\): for each cluster j'th, for each member in that cluster, swap the current center with that member and calculate the sum of distance from all member in the cluster to that new center. Pick the member that after swapping with the current center, have the smallest sum of distances. 
4. Repeat step 2 and step 3 until convergence 

### 5.3. Properties
1. It is \\( O (n^2)\\) in complexity, since it needs to calculate all the pairwise distances. So it is (much) slower than K-Means. 
2. It works naturally with any distance metrics. 
3. Note again that the center (medoid) must be one of the point in the dataset, unlike K-Means which can be anywhere 
4. It is more robust to outlier than K-Means

## 6. Hierarchical Clustering
<div class="imgcap">
<div>
<img src="/assets/unsupervised_learning/hc.png">
</div>
<div class="thecap">Illustration of Hierarchical Clustering. Image Credit: Wikipedia </div>
</div>
### 6.1. Algorithm
We present here the bottom-up approach. One can form a tree along the process. 

1. Define a notion of distance between a cluster and another cluster, for example use the minimum distance between any a point from this cluster to another point from the other cluster. 
2. Starts with \\( n\\) clusters where each point is its own cluster
3. Repeat \\( n - 1\\) times : pick two clusters that are closest to each other and merge them.  

### 6.2. Properties
1. It is very slow \\( O (n ^ 3)\\) with the naive implementation. It can be reduced to \\( O (n ^ 2)\\)
2. The top-down approach, where one starts with one cluster and gradually breaking down cluster, have exponential complexity. 
3. It is useful for visualizing and interpreting features when doing supervise learning. One can produce a dendogram of the features to see which one are correlated. 
4. It perform clustering for all \\( k \\) - number of cluster - at the same time. 

## 6. Gaussian Mixture Models
GMM can be thought of as a soft version of K-Means. In K-Means each point either belong to this cluster or some other cluster. In GMM, each point has a probability \\( \pi _i \\) of being to cluster \\( i\\)'th. GMM can be solved somewhat similarly to K-Means. We won't go into detail GMM here, and will cover it when we talk about Bayseian models, graphical models. 
