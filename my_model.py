import numpy as np
import os
import tensorflow as tf
from my_data_utils import minibatches, pad_sequences, get_tuples, get_logger



class Seq_Model(object):
    def __init__(self, configuration, embeddings, ntags, logger=None):
        """
        Bi lstm model for pos tagging
        """
        self.configuration  = configuration
        self.embeddings = embeddings
        self.ntags      = ntags

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.logger = logger


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        """
        # perform padding of the given data

        word_ids, sequence_lengths = pad_sequences(words)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }


        if labels is not None:
            labels, _ = pad_sequences(labels)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_summary(self, sess):
        # tensorboard
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.configuration.output_path, sess.graph)


    def build(self):
        # (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],name="word_ids")
        # (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")
        # (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],name="word_lengths")
        # (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],name="labels")
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # Add word embeddings
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                trainable=self.configuration.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                name="word_embeddings")

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

        # Build LSTM with lstm cell (bi lstm)
        with tf.variable_scope("bi-lstm"):

            lstm_cell = tf.contrib.rnn.LSTMCell(self.configuration.hidden_size)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                lstm_cell, self.word_embeddings, sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        # Decoding - find the sequence with highest probability
        # Matrix W
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2*self.configuration.hidden_size, self.ntags],
                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.configuration.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

        # We are gonna use cross-entropy loss, the loss is -log(p(y)) where y is
        # the correct sequence of tags and the probability is given by the crf

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        self.logits, self.labels, self.sequence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()


    def predict_batch(self, sess, words):
        """
        Prediction
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        viterbi_sequences = []
        logits, transition_params = sess.run([self.logits, self.transition_params],
                feed_dict=fd)

        # iterate over the phrases
        for logit, sequence_length in zip(logits, sequence_lengths):
            # keep only the valid time steps
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                            logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences, sequence_lengths




    def run_epoch(self, sess, train, dev, tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        """
        nbatches = (len(train) + self.configuration.batch_size - 1) / self.configuration.batch_size

        for i, (words, labels) in enumerate(minibatches(train, self.configuration.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.configuration.lr, self.configuration.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
        print "run evaluate"
        acc, f1 = self.run_evaluate(sess, dev, tags)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
        return acc, f1


    def run_evaluate(self, sess, test, tags):
        """
        Evaluates performance

        Returns:
            accuracy
            f1 score
        """

        accs = []
        correct = 0.
        total_correct = 0.
        total_preds = 0.

        for words, labels in minibatches(test, self.configuration.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += map(lambda (a, b): a == b, zip(lab, lab_pred))

                lab_chunks = set(get_tuples(lab, tags))
                lab_pred_chunks = set(get_tuples(lab_pred, tags))
                correct += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        # f1 score
        p = correct / total_preds if correct > 0 else 0
        r = correct / total_correct if correct > 0 else 0
        f1 = 2 * p * r / (p + r) if correct > 0 else 0

        acc = np.mean(accs)
        return acc, f1


    def train(self, train, dev, tags):
        """
        Training
        """
        best_score = 0
        saver = tf.train.Saver()

        # early stopping
        nepoch_no_imprv = 0

        with tf.Session() as sess:
            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)


            for epoch in range(self.configuration.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.configuration.nepochs))

                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)

                # decay
                self.configuration.lr *= self.configuration.lr_decay

                # early stopping
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.configuration.model_output):
                        os.makedirs(self.configuration.model_output)
                    saver.save(sess, self.configuration.model_output)
                    best_score = f1
                    self.logger.info("score is better")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= 3:
                        self.logger.info("early stopping {} epochs".format(
                                        nepoch_no_imprv))
                        break


    def evaluate(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing")
            saver.restore(sess, self.configuration.model_output)
            acc, f1 = self.run_evaluate(sess, test, tags)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
