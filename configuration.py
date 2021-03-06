class configuration():
    dim = 100
    max_iter = None
    lowercase = True
    train_embeddings = True
    nepochs = 1
    dropout = 0.5
    batch_size = 20
    lr = 0.01
    lr_decay = 0.05
    nepoch_no_imprv = 3
    hidden_size = 200
    crf = True
    glove_filename = "glove.6B/glove.6B.{}d.txt".format(dim)
    trimmed_filename = "glove.6B.{}d.trimmed.npz".format(dim)
    dev_filename = "dev_rnn.txt"
    test_filename = "test_rnn.txt"
    train_filename = "train_rnn.txt"
    words_filename = "words.txt"
    tags_filename = "tags.txt"
    chars_filename = "chars.txt"
    train_x = "train_x.csv"
    train_y = "train_y.csv"
    dev_x = "dev_x.csv"
    dev_y = "dev_y.csv"
    test_x = "test_x.csv"
    output_path = "results/crf/"
    model_output = "model.weights/"
    log_path = "log.txt"
