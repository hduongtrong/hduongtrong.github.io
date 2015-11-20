---
layout: post
comments: true
title:  "Deep Reinforcement Learning"
date:   2015-10-20 01:28:27
excerpt: Summary of the UC Berkeley CS294 Deep Reinforcement Learning
mathjax: true
---

Deep Reinforcement Learning is an exciting new field that encompasses many
different fields: computer science, optimal control, statistics, machine
learning, and so on. Its application are numerous.

## Policy Gradient
**Definition** A Markov Decision Process contains

* \\( \pi : \mathcal{S} \rightarrow \Delta(\mathcal(A)) \\), the stochastic policy. 
* \\( \eta(\pi) = \mathbb{E} \left[ R_0 + \gamma R\_1 + \gamma^2 R\_2 +  
   ... \right] \\)
* \\( p: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R} \\), the state transition probability
* \\( \mu : \mathcal{S} \rightarrow \mathbb{R} \\), the probability distribution over the initial state, \\( s_0 \\)
* \\( \theta \in \mathbb{R}^n \\), a vector of parameter that parameterizes the stochastic policy \\(\pi\\)

A policy gradient algorithm then calculate \\( \nabla\_ {\theta} \eta (\theta) \\), and make proceed as a standard gradient ascent algorithm. We approximate the gradient using Monte Carlo estimation, since we don't have access to the underlying probability distribution. 

**Theorem** Monte Carlo estimation. Let \\( X : \Omega \rightarrow \mathbb{R} ^ n \\) be a random variable with probability distribution \\(q\\), and \\(f : \mathbb {R}^n \rightarrow \mathbb{R} \\). Then 
	$$ \frac{\partial}{\partial \theta } \mathbb {E} \left[ f(X) \right] = 
		\mathbb{E} \left[ f(X) \frac{\partial}{\partial \theta } 
		\log q(X;\theta) \right]$$

Armed with this theorem, we can use a sample average to estimate the expectation. 

**Proposition** 

```python
for i in xrange(10):
	if 
```