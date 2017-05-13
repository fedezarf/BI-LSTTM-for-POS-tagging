# BI-LSTTM-for-POS-tagging

The RNN Architecture

The architecture used is a BI-LSTM network with CRF (Conditional Random Field) and a Viterbi decoding to find the best sequence. The architecture is composed of:

Word embeddings (we experimented with different embedding dimensions) with dropout
Forward and Backward LSTM layer (experimented with different hidden sizes) with dropout
CRF Layer

![alt text](https://guillaumegenthial.github.io/assets/bi-lstm.png) 

This architecture was inspired by the scientific paper: https://arxiv.org/pdf/1603.01354.pdf by Xuezhe Ma and Eduard Hovy, even though I decided to not implement the character embedding due to the expensive computation of the training. 

The network behave like this: we run a bi-LSTM over the sequence of word vectors and obtain another sequence of vectors (the concatenation of the two hidden states). Each word is associated to a vector that captures information from the meaning of the word and its context. 
Then we use CRF to make the prediction at sentence level. Like the last assignment, we want to predict the probability of a sequence of POS tags for a given sentence. To compute the CRF we take a matrix W and a vector of scores. The CRF is better than softmax because the softmax makes local choices. 

Given a sequence of words, a sequence of score vectors and a sequence of tags a linear-chain CRF defines a global score where we have a transition matrix and vectors of scores that capture the cost of beginning or starting with a given tag. We use the matrix to capture linear dependencies between tagging decisions. Then, we can use the dynamic programming of the viterbi algorithm to find the argmax. In conclusion, the final step of a linear chain CRF is to apply a softmax to the scores of all possible sequences to get the probability of a given sequence of tags. 

The objective function of our neural network is the cross-entropy loss -log(P(y)) where y is the correct sequence of tags and its probability is given by the CRF softmax. Using the Tensorflow library is easy to compute the loss of the CRF.

For the padding, we chose to do it dynamically: pad to the maximum length in the batch. Thus, sentence length and word length will depend on the batch. (see code pad_sequences).



