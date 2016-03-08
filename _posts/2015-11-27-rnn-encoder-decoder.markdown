---
layout: post
comments: true
title:  "Sequence to Sequence Model"
date:   2015-11-27 
excerpt: Also known as Neural Encoder-Decoder model
mathjax: true
---

## Credits

* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf). Kyunghyun Cho, Bart van Merrienboer Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares Holger Schwenk, Yoshua Bengio. [1]
* [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://arxiv.org/pdf/1409.1259v2.pdf). Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau.
* [Grammar as a Foreign Language](http://arxiv.org/pdf/1412.7449v3.pdf). Oriol Vinyals, Lukasz Kaiser, 
Terry Koo, Slav Petrov, Ilya Sutskever. 
* [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/pdf/1409.0473v6.pdf). Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio. 
* Code Website: [Tensorflow seq2seq](http://www.tensorflow.org/tutorials/seq2seq/index.html#sequence-to-sequence-models)

## The Core Model

<div class="imgcap">
<div>
<img src="/assets/seq2seq/seq2seq.png" style="width: 500px;">
</div>
<div class="thecap"> Sequence to Sequence Model </div>
</div>

First we define the data, which include encoder input, decoder input, decoder output, encoder hidden states, and decoder hidden states. Here \\( S \\) denotes the dummy START symbol. 
 
1. Encoder input sequence \\( X _ 1, X _ 2, X _ 3  \in \mathbb{R} ^ p \\), 
2. Encoder hidden states be \\( H _ 0, H _ 1, H _ 2, H _ 3 \in \mathbb{R} ^ s, H _ 0 = 0 \\). 
3. Decoder input sequence \\( S, Y _ 1, Y _ 2 \in \mathbb{R} ^ {q} \\)
4. Decoder output sequence \\( Y _ 1, Y _ 2, Y _ 3 \in \mathbb{R} ^ {q} \\)
5. Decoder hidden states be \\( K _ 0, K _ 1, K _ 2 \in \mathbb{R} ^ t \\)

Let \\( \sigma \\) be the sigmoid function, we now define the parameter for our model: 

1. Weight matrix that connects encoder input to encoder hidden state be \\( W _ x \in \mathbb{R} ^ {p \times s}\\). 
2. Weight matrix that connects this encoder hidden state to the next hidden state be \\( W _ h \in \mathbb{R} ^ {s \times s}\\). 
3. Weight matrix that connects decoder input to decoder hidden state \\( W _ y \in \mathbb{R} ^ {q \times }\\)
4. Weight matrix that connects this decoder hidden state to the next hidden state \\( W _ k \in \mathbb{R} ^ {t \times t}\\)
5. Weight matrix that connects this decoder hidden state to the decoder hidden output \\( W _ z \in \mathbb{R} ^ {t \times q}\\)

We now model the probability of \\( Y _ i\\) as a function of \\( X _ i 's\\) through the intermediates \\( H _ i 's , K _ i 's \\)

$$
	\begin{equation}
	\begin{split}
	    H _ 1 &= \sigma (W _ x X _ 1 + W _ h H _ 0) \\\\
		H _ 2 &= \sigma (W _ x X _ 2 + W _ h H _ 1) \\\\
		H _ 3 &= \sigma (W _ x X _ 3 + W _ h H _ 2) \\\\
		K _ 1 &= \sigma (W _ y S     + W _ k H _ 3) \\\\
		K _ 2 &= \sigma (W _ y Y _ 1 + W _ k K _ 1) \\\\
		K _ 3 &= \sigma (W _ y Y _ 2 + W _ k K _ 2) \\\\
		\hat{Y} _ 1 &= \sigma (W _ z K _ 1) 				\\\\
		\hat{Y} _ 2 &= \sigma (W _ z K _ 2) 				\\\\
		\hat{Y} _ 3 &= \sigma (W _ z K _ 3) 				\\\\
	\end{split}
	\end{equation}
$$

Now as we can see, \\( Y _ 1\\) is a function (composition of functions) of \\(X _ 1, X _ 2, X _ 3\\). \\( Y _ 2 \\) in addition also depends on \\( Y _ 1\\), and so on. The loss function in the case of classification, if we use the multinomial loss, is then just

$$
	\begin{equation}
	\begin{split}
    	L(W _ x, W _ h, W _ y, W _ k, W _ z) &= \sum _ {i = 1} ^ 3 f(Y _ i , \hat{Y _ i }) \\\\
		f(Y, \hat{Y}) 						 &= \sum Y[j] \log \hat{Y}[j], \\\\
	\end{split}
	\end{equation}
$$
for in the last equation, we use array notation, \\( Y[j]\\) means the j'th element in vector \\( Y \\). 

This loss function is only a function of the data and the 5 parameter matrix. We can calculate the gradient, and use gradient descent or mini-batch stochastic gradient descent to optimize it. 

## Improvements

Above is the core of a sequence to sequence model. In practice, we should use

1. LSTM cell instead of a vanilla RNN cell. What this means is instead of having \\(H _ t = \sigma (W _ x \times X + W _ h H _ {t - 1})\\), we have a more complicated function here. 
2. Multiple layers instead of just one hidden layer as we have
3. Add a bias term in addition to the weight \\( W \\) term
4. For variable size output, we have an additional END symbol in addition to the START symbol. We cut the output when the model predict an END. 

Some implementation nodes

The code [Seq2seq in Tensorflow](https://github.com/hduongtrong/ScikitFlow/blob/master/seq2seq.py)