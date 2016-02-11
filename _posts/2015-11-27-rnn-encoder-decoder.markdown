---
layout: post
comments: true
title:  "RNN Encoder Decoder - Neural Translation Machine"
date:   2015-11-27 
excerpt: Summary of RNN Encoder-Decoder approach. Neural Translation Machine
mathjax: true
---

## Credits

* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf). Kyunghyun Cho, Bart van Merrie ̈nboer Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares Holger Schwenk, Yoshua Bengio. [1]
* [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://arxiv.org/pdf/1409.1259v2.pdf). Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau.
* [Grammar as a Foreign Language](http://arxiv.org/pdf/1412.7449v3.pdf). Oriol Vinyals, Lukasz Kaiser, 
Terry Koo, Slav Petrov, Ilya Sutskever. 
* [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/pdf/1409.0473v6.pdf). Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio. 
* Code Website: [Tensorflow seq2seq](http://www.tensorflow.org/tutorials/seq2seq/index.html#sequence-to-sequence-models)

## RNN Encoder-Decoder

The paper [1] already has a succint explaination of this model. I copy most of stuff over here. 