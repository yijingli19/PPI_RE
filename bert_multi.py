from keras.callbacks import TensorBoard
from keras.layers import *
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import *
from keras.optimizers import *
from sklearn import metrics
import pandas as pd
import re
import os
import numpy as np


def cnn_model(embedding_matrix, x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    main_input1 = Input(shape=(30,), dtype='float32', name='input1')
    main_input2 = Input(shape=(30,), dtype='float32', name='input2')
    main_input3 = Input(shape=(30,), dtype='float32', name='input3')
    main_input4 = Input(shape=(30,), dtype='float32', name='input4')
    main_input5 = Input(shape=(30,), dtype='float32', name='input5')
    main_input6 = Input(shape=(100, 768), dtype='float32', name='input6')  # token embedding
    # 词嵌入（使用预训练的词向量）
    emb1 = Embedding(len(vocab) + 1, 200, input_length=30, weights=[embedding_matrix[0]], trainable=False)(main_input1)
    emb2 = Embedding(len(vocab) + 1, 200, input_length=30, weights=[embedding_matrix[0]], trainable=False)(main_input2)
    emb3 = Embedding(len(vocab) + 1, 200, input_length=30, weights=[embedding_matrix[0]], trainable=False)(main_input3)
    emb4 = Embedding(len(vocab) + 1, 200, input_length=30, weights=[embedding_matrix[0]], trainable=False)(main_input4)
    emb5 = Embedding(len(vocab) + 1, 200, input_length=30, weights=[embedding_matrix[0]], trainable=False)(main_input5)
    embed = Concatenate(axis=1)([emb1, emb2, emb3, emb4, emb5])
    # w2v
    cnn_w2v_1 = Conv1D(200, 3, strides=1, activation='relu')(embed)
    cnn_w2v_1 = MaxPooling1D(pool_size=15)(cnn_w2v_1)
    cnn_w2v_2 = Conv1D(200, 4, strides=1, activation='relu')(embed)
    cnn_w2v_2 = MaxPooling1D(pool_size=15)(cnn_w2v_2)
    cnn_w2v_3 = Conv1D(200, 5, strides=1, activation='relu')(embed)
    cnn_w2v_3 = MaxPooling1D(pool_size=15)(cnn_w2v_3)
    cnn_w2v = concatenate([cnn_w2v_1, cnn_w2v_2, cnn_w2v_3], axis=-1)
    # bert
    cnn_bert_1 = Conv1D(200, 3, strides=1, activation='relu')(main_input6)
    cnn_bert_1 = MaxPooling1D(pool_size=15)(cnn_bert_1)
    cnn_bert_2 = Conv1D(200, 4, strides=1, activation='relu')(main_input6)
    cnn_bert_2 = MaxPooling1D(pool_size=15)(cnn_bert_2)
    cnn_bert_3 = Conv1D(200, 5, strides=1, activation='relu')(main_input6)
    cnn_bert_3 = MaxPooling1D(pool_size=15)(cnn_bert_3)
    cnn_bert = concatenate([cnn_bert_1, cnn_bert_2, cnn_bert_3], axis=-1)

    combined = Concatenate(axis=1)([cnn_w2v, cnn_bert])
    flat = Flatten()(combined)
    main_output = Dense(2, activation="sigmoid")(flat)
    model = Model(inputs=[main_input1, main_input2, main_input3, main_input4, main_input5, main_input6], outputs=main_output)
    model.summary()

    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="log")  # tensorboard的示例
    callbacks_tensor = [tensorboard]
    model.fit([x_train_padded_seqs[0], x_train_padded_seqs[1], x_train_padded_seqs[2], x_train_padded_seqs[3], x_train_padded_seqs[4], x_train_padded_seqs[5]], y_train, batch_size=16, epochs=64, callbacks=callbacks_tensor, shuffle=True)
    model_save_path = "./output/multi_bert.h5"
    model.save(model_save_path)
    model = load_model(model_save_path)
    result = model.predict([x_test_padded_seqs[0], x_test_padded_seqs[1], x_test_padded_seqs[2], x_test_padded_seqs[3], x_test_padded_seqs[4], x_test_padded_seqs[5]])  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    y_test = np.argmax(y_test, axis=1)  # 获得最大概率对应的标签
    y_test = list(map(str, y_test))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    performance(y_predict, y_test)
    # print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='binary'))


def load_word2vec(vocab):
    data_dir = './embedding'
    # embedding_path_list = ['PMC-w2v', 'PubMed-w2v', 'wikipedia-pubmed-and-PMC-w2v', 'pubmed2018_w2v_200D']
    embedding_path_list = ['wikipedia-pubmed-and-PMC-w2v']
    embedding_matrix_list = []
    for path in embedding_path_list:
        embedding_matrix_path = os.path.join(data_dir, path + '.npy')
        if os.path.exists(embedding_matrix_path):
            print("Load word2vec file {0}\n".format(embedding_matrix_path))
            embedding_matrix = np.load(embedding_matrix_path)
        else:
            # initial matrix with random uniform
            embedding_matrix = np.zeros((len(vocab) + 1, 200))
            embedding_path = os.path.join(data_dir, path + '.bin')
            # load any vectors from the word2vec
            print("Load word2vec file {0}\n".format(embedding_path))
            embeddings_index = dict()
            with open(embedding_path, "rb") as f:
                header = f.readline()
                vocab_size, layer_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer_size
                for line in range(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1).decode('latin-1')
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    idx = vocab.get(word)
                    if idx is not None:
                        embeddings_index[word] = np.frombuffer(f.read(binary_len), dtype='float32')
                    else:
                        f.read(binary_len)
            for word, i in vocab.items():
                # 在词向量索引字典中查询单词word的词向量
                embedding_vector = embeddings_index.get(word)
                # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            np.save(embedding_matrix_path, embedding_matrix)
        embedding_matrix_list.append(embedding_matrix)
    return embedding_matrix_list


def process_file(filename):
    class2label = {'False': 0, 'True': 1}
    data = []
    lines = [line.strip() for line in open(filename)]
    for line in lines:
        inform = line.split('\t')
        relation = inform[3]
        # e1_type = inform[1]
        # e2_type = inform[3]
        # get sentence
        sentence = inform[2]
        # string to list
        # tokens = nltk.word_tokenize(sentence)
        sentence = sentence.replace('<*>', '_e11_')
        sentence = sentence.replace('</*>', '_e12_')
        sentence = sentence.replace('<#>', '_e21_')
        sentence = sentence.replace('</#>', '_e22_')
        sentence = clean_str(sentence)
        tokens = sentence.split(' ')
        e1_start = tokens.index("e11")
        e1_end = tokens.index("e12")
        e2_start = tokens.index("e21")
        e2_end = tokens.index("e22")
        # get 5 channel and their position
        e1_p1, e1_p2, e2_p1, e2_p2, l_p1, l_p2, m_p1, m_p2, r_p1, r_p2 = '', '', '', '', '', '', '', '', '', ''
        entity1 = ' '.join(tokens[e1_start:e1_end + 1])
        entity1 = entity1.replace('<*> ', '').replace(' </*>', '')

        entity2 = ' '.join(tokens[e2_start:e2_end + 1])
        entity2 = entity2.replace('<#> ', '').replace(' </#>', '')

        left = ' '.join(tokens[:e1_start])
        if left != '':
            for word in left.split(' '):
                l_p1 += str(tokens.index(word) - e1_start) + ' '
                l_p2 += str(tokens.index(word) - e2_start + 2) + ' '
        else:
            l_p1 = str(1)
            l_p2 = str(e2_start - 2)

        middle = ' '.join(tokens[e1_end + 1:e2_start])
        if middle != '':
            for word in middle.split(' '):
                m_p1 += str(tokens.index(word) - e1_end) + ' '
                m_p2 += str(tokens.index(word) - e2_start) + ' '
        else:
            m_p1 = str(-1)
            m_p2 = str(1)

        right = ' '.join(tokens[e2_end + 1:])
        if right != '':
            for word in right.split(' '):
                r_p1 += str(tokens.index(word) - e1_end - 2) + ' '
                r_p2 += str(tokens.index(word) - e2_end) + ' '
        else:
            r_p1 = str(e2_end - e1_end + 1 - 2)
            r_p2 = str(1)

        # if channel is empty, replace with '@'
        if left == '':
            left = '@'
        if middle == '':
            middle = '@'
        if right == '':
            right = '@'

        data.append([id, entity1, entity2, left, middle, right, relation])
    print(filename)
    # data to some 2-d structure like excel
    df = pd.DataFrame(data=data, columns=["id", "entity1", "entity2", "left", "middle", "right", "relation"])
    # label
    df['label'] = [class2label[r] for r in df['relation']]
    y = df['label']
    # Text Data
    x_text1 = df['left'].tolist()
    x_text2 = df['entity1'].tolist()
    x_text3 = df['middle'].tolist()
    x_text4 = df['entity2'].tolist()
    x_text5 = df['right'].tolist()

    # to 1-d, like flat
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return (x_text1, x_text2, x_text3, x_text4, x_text5), labels


def clean_str(text):
    # text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=*]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\\", " \\ ", text)
    text = re.sub(r"\*", " * ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def performance(pre_list, test_list):
    true_num = 0  # 预测结果中真正的正例数目
    pre_true = 0  # 预测结果中认为是正例的数目
    test_true_num = 0  # 测试集中的正例数目

    for test_node in test_list:
        if test_node == '1':
            test_true_num += 1

    for pre_node in pre_list:
        if pre_node == '1':
            pre_true += 1

    for test_node_child, pre_node_child in zip(test_list, pre_list):
        if test_node_child == pre_node_child and test_node_child == '1' and pre_node_child == '1':
            true_num += 1

    print("-------------------------*", type, "*---------------------------------")
    print("|测试集中的正例的数目: " + str(test_true_num))
    print("|预测结果中认为是正例的数目:" + str(pre_true))
    print("|预测为正例的结果中真正的正例数目:" + str(true_num))
    P = 0 if pre_true == 0 else 100. * true_num / pre_true
    R = 0 if test_true_num == 0 else 100. * true_num / test_true_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("|Precision: %.2f" % (P), "%")
    print("|Recall: %.2f" % (R), "%")
    print("|F1: %.2f" % (F), "%")
    print()


if __name__ == '__main__':
    data_root = './data/HPRD50'
    train_path = os.path.join(data_root, 'train.txt')
    test_path = os.path.join(data_root, 'test.txt')
    x_train, y_train = process_file(train_path)
    x_test, y_test = process_file(test_path)
    print("train/test：", len(y_train), "/", len(y_test), '\n')
    tokenizer = Tokenizer()  # 创建一个Tokenizer对象
    # 将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer.fit_on_texts(x_train[0] + x_train[1] + x_train[2] + x_train[3] + x_train[4] + x_test[0] + x_test[1] + x_test[2] + x_test[3] + x_test[4])
    vocab = tokenizer.word_index  # 得到每个词的编号
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train0_word_ids = tokenizer.texts_to_sequences(x_train[0])
    x_train1_word_ids = tokenizer.texts_to_sequences(x_train[1])
    x_train2_word_ids = tokenizer.texts_to_sequences(x_train[2])
    x_train3_word_ids = tokenizer.texts_to_sequences(x_train[3])
    x_train4_word_ids = tokenizer.texts_to_sequences(x_train[4])

    x_test0_word_ids = tokenizer.texts_to_sequences(x_test[0])
    x_test1_word_ids = tokenizer.texts_to_sequences(x_test[1])
    x_test2_word_ids = tokenizer.texts_to_sequences(x_test[2])
    x_test3_word_ids = tokenizer.texts_to_sequences(x_test[3])
    x_test4_word_ids = tokenizer.texts_to_sequences(x_test[4])

    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train0_padded_seqs = pad_sequences(x_train0_word_ids, maxlen=30)  # [5799,50]
    x_train1_padded_seqs = pad_sequences(x_train1_word_ids, maxlen=30)
    x_train2_padded_seqs = pad_sequences(x_train2_word_ids, maxlen=30)
    x_train3_padded_seqs = pad_sequences(x_train3_word_ids, maxlen=30)
    x_train4_padded_seqs = pad_sequences(x_train4_word_ids, maxlen=30)

    x_test0_padded_seqs = pad_sequences(x_test0_word_ids, maxlen=30)
    x_test1_padded_seqs = pad_sequences(x_test1_word_ids, maxlen=30)
    x_test2_padded_seqs = pad_sequences(x_test2_word_ids, maxlen=30)
    x_test3_padded_seqs = pad_sequences(x_test3_word_ids, maxlen=30)
    x_test4_padded_seqs = pad_sequences(x_test4_word_ids, maxlen=30)

    embedding_matrix = load_word2vec(vocab)  # [len(vocab), 200]
    # bert_train_emb
    train_emb_path = os.path.join(data_root, 'train_emb.npy')
    print("Load bert_train_emb {0}\n".format(train_emb_path))
    train_emb = np.load(train_emb_path)  # sent_emb:[,768],  token_emb: [len(dara)*100,768]
    # bert_test_emb
    test_emb_path = os.path.join(data_root, 'test_emb.npy')
    print("Load bert_test_emb {0}\n".format(test_emb_path))
    test_emb = np.load(test_emb_path)

    cnn_model(embedding_matrix,
              [x_train0_padded_seqs, x_train1_padded_seqs, x_train2_padded_seqs, x_train3_padded_seqs, x_train4_padded_seqs, train_emb],
              y_train,
              [x_test0_padded_seqs, x_test1_padded_seqs, x_test2_padded_seqs, x_test3_padded_seqs, x_test4_padded_seqs, test_emb],
              y_test)
