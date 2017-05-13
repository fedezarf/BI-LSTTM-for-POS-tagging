import numpy as np
import os
from  configuration import configuration
import logging

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class MY_Dataset(object):
    """
    Proces the dataset and create an iterator
    """
    def __init__(self, filename, processing_word=None, processing_tag=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        yield words, tags
                        words, tags = [], []
                else:
                    word, tag = line.split(' ')
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_vocabs(datasets):
    vocab_words = set()
    vocab_tags = set()

    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)

    vocab_tags.update("O")

    return vocab_words, vocab_tags


def get_embed_vocab(filename):
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)

    return vocab

def write_vocab(vocab, filename):
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)

def load_vocab(filename):
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_embed_vectors(vocab, embed_filename, res_filename, dim):
    """
    Saves embedding vectors in numpy array
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(embed_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = map(float, line[1:])
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(res_filename, embeddings=embeddings)


def get_embed_vectors(filename):
    with open(filename) as f:
        return np.load(f)["embeddings"]


def get_processing_word(vocab_words=None,lowercase=False):
    """
    process word
    """
    def f(word):

        # preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        return word

    return f


def pad_sequences(sequences):
    """
    Returns:
        a list of list where each sublist has same length
    """

    max_length = max(map(lambda x : len(x), sequences))
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [0]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """data: generator of (sentence, tags) tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_pos_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('-')[-1]


def get_tuples(seq, tags):
    """
        seq: [4, 4, 0, 0, ...] sequence of labels (pos tags)
        tags: dict["O"] = 4
    Returns:
        result = [("VBZ", 0, 2), ("NNS", 3, 4)]
    """

    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.iteritems()}

    tuples = []
    tuple_type, tuple_start = None, None

    for i, tup in enumerate(seq):

        if tup == default and tuple_type is not None:

            chunk = (tuple_type, tuple_start, i)
            tuples.append(chunk)
            tuple_type, tuple_start = None, None


        elif tup != default:
            tup_chunk_type = get_pos_type(tup, idx_to_tag)
            if tuple_type is None:
                tuple_type, tuple_start = tup_chunk_type, i
            elif tup_chunk_type != tuple_type:
                chunk = (tuple_type, tuple_start, i)
                tuples.append(chunk)
                tuple_type, tuple_start = tup_chunk_type, i
        else:
            pass

    if tuple_type is not None:
        chunk = (tuple_type, tuple_start, len(seq))
        tuples.append(chunk)
    return tuples

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
