from configuration import configuration
from my_data_utils import MY_Dataset, get_vocabs, UNK, NUM, \
    get_embed_vocab, write_vocab, load_vocab,  \
    export_embed_vectors, get_processing_word, get_embed_vectors, \
    get_logger
from my_model import Seq_Model
import pandas as pd
import logging

def test_util(pd_):
    dev_x_phrases = []
    for i,row in pd_.iterrows():
        if row['word'] == '-DOCSTART-':
            new_sentence = []
            c = i+1
            if c == len(pd_):
                break
            while pd_.iloc[c]['word'] != '-DOCSTART-':
                new_sentence.append(pd_.iloc[c]['word'])
                c += 1
                if c == len(pd_):
                    break
            dev_x_phrases.append(new_sentence)

            i = c

    return dev_x_phrases

def build_data(configuration):
    """
    Procedure to build data
        creates vocab files from the datasets
        creates a npz embedding file from embed vectors
    """

    train_x = pd.read_csv(configuration.train_x)
    train_y = pd.read_csv(configuration.train_y)
    train_x["tag"] = train_y["tag"]
    df = train_x.drop('id', 1)
    
    df.to_csv(configuration.train_filename, header=None, index=None, sep=' ', mode='a')

    dev_x = pd.read_csv(configuration.dev_x)
    dev_y = pd.read_csv(configuration.dev_y)
    dev_x["tag"] = dev_y["tag"]
    df_dev = dev_x.drop('id', 1)
    df_dev.to_csv(configuration.dev_filename, header=None, index=None, sep=' ', mode='a')

    test_x = pd.read_csv(configuration.test_x)
    test = test_util(test_x)

    processing_word = get_processing_word(lowercase=configuration.lowercase)

    # Generators
    dev   = MY_Dataset(configuration.dev_filename, processing_word)
    test  = MY_Dataset(configuration.test_filename, processing_word)
    train = MY_Dataset(configuration.train_filename, processing_word)

    print "Generators building done"
    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev])
    vocab_glove = get_embed_vocab(configuration.glove_filename)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)


    # Save vocab
    write_vocab(vocab, configuration.words_filename)
    write_vocab(vocab_tags, configuration.tags_filename)

    # Embed vectors
    vocab = load_vocab(configuration.words_filename)
    export_embed_vectors(vocab, configuration.glove_filename, configuration.trimmed_filename, configuration.dim)


if __name__ == "__main__":
    build_data(configuration)

    print "Data built"
    # load vocabs
    vocab_words = load_vocab(configuration.words_filename)
    vocab_tags  = load_vocab(configuration.tags_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, lowercase=configuration.lowercase)
    processing_tag  = get_processing_word(vocab_tags, lowercase=False)

    # get pre trained embeddings
    embeddings = get_embed_vectors(configuration.trimmed_filename)

    # create dataset
    dev   = MY_Dataset(configuration.dev_filename, processing_word,
                    processing_tag)
    test  = MY_Dataset(configuration.test_filename, processing_word,
                    processing_tag)
    train = MY_Dataset(configuration.train_filename, processing_word,
                    processing_tag)

    # get logger
    logger = get_logger(configuration.log_path)

    print "Everything was loaded"

    # build model
    model = Seq_Model(configuration, embeddings, ntags=len(vocab_tags), logger=logger)
    model.build()

    print "Model initialized"

    print "Begin training"
    model.train(train, dev, vocab_tags)

    print "Training finished, predicting..."

    final_pred = []

    idx_to_tag = {idx: tag for tag, idx in vocab_tags.iteritems()}
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, configuration.model_output)
        for phrase in test:
            words_raw = phrase
            words = map(processing_word, words_raw)
            if type(words[0]) == tuple:
                words = zip(*words)
            pred_ids, _ = model.predict_batch(sess, [words])
            preds = map(lambda idx: idx_to_tag[idx], list(pred_ids[0]))
            final_pred.append(preds)

    print "Test prediction finished"

    i = 0
    sub = []
    for elem in final_pred:
        sub.append("O")
        for t in elem:
            sub.append(t)

    sub_final = []
    for i,elem in enumerate(sub):
        sub_final.append((i,elem))

    sub_pd = pd.DataFrame(sub_final, columns=['id','tag'])

    sub_pd.to_csv("hw4_test_final_bilstm.csv", index=False)

    print "Test submission file generated"
