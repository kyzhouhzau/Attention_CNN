#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Time:2018/8/25
@Author:zhoukaiyin
"""
import numpy as np
import pickle
import gensim
import os
import tensorflow as tf
from collections import namedtuple

FLAGS = tf.flags.FLAGS

Data_example = namedtuple("data_example", "label,entity1,entity2,sentence")
Entity_position = namedtuple("Entity_position", "first last")

def load_raw_data(datapath):
    """
    将data的每一行数据以Data_example的格式保存。
    :return: data
    """
    data = []
    with open(datapath, 'r') as rf:
        for line in rf:
            line_lis = line.strip().split()
            if len(line_lis) > FLAGS.max_len:
                FLAGS.max_len = len(line_lis)
            sentence = line_lis[5:]

            entity1 = Entity_position(int(line_lis[1]), int(line_lis[2]))
            entity2 = Entity_position(int(line_lis[3]), int(line_lis[4]))
            line_example = Data_example(line_lis[0], entity1, entity2, sentence)
            data.append(line_example)
    return data


def vocab2id(train_data, test_data, vocabfile):
    """
    将训练数据与测试数据中所有出现的单词去重后映射到id
    :param train_data:
    :param test_data:
    :param vocabfile:
    :return: 将vocab2id的映射写入磁盘
    """
    vocab = []
    for example in train_data + test_data:
        for w in example.sentence:
            vocab.append(w)
    words = list(set(vocab))
    vocab2id = {w: i for i, w in enumerate(words, 1)}
    vocab2id["<pad>"] = 0
    if not os.path.exists(vocabfile):
        with open(vocabfile, 'wb') as wf:
            pickle.dump(vocab2id, wf)


def _load_vocab(vocabfile):
    with open(vocabfile, 'rb') as rf:
        vocab2id = pickle.load(rf)
    return vocab2id


def embed_trim(vocabfile, embedtrimfile):
    """
    由于词向量本身较大，这里通过修剪将不必要的去掉，只保留与vocabfile中对应的部分。
    :param vocabfile:
    :param embedtrimfile:google pre_trained word2vec
    :return:将修建后的向量存入磁盘供后面model的调用。
    """
    trimedembed = []
    if not os.path.exists(embedtrimfile):
        if FLAGS.word_dim == 200:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove200.txt")
        elif FLAGS.word_dim == 50:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove50.txt")
        # pad_embed =[0]*model.vector_size
        vocab2id = _load_vocab(vocabfile)
        id2vocab = {i: w for w, i in vocab2id.items()}
        ebed_vocab = model.vocab.keys()
        count = 0
        for i in range(len(id2vocab)):
            w = id2vocab[i]
            if w in ebed_vocab:
                trimedembed.append(model[w])
            else:
                word_lis = w.split('_')
                try:
                    for m in word_lis:
                        np_embed = np.zeros([model.vector_size])
                        w_embed = model[m]
                        np_embed += w_embed
                        trimedembed.append(list(np_embed))
                except KeyError:
                    count += 1
                    npdata = np.random.normal(0, 0.1, [model.vector_size])
                    trimedembed.append(list(npdata))
        embed = np.asarray(trimedembed).astype(np.float32)
        print("在构建---单词---词向量的时候有{}个单词没有找到词向量!".format(count))
        np.save(embedtrimfile, embed)



def _load_embed(embedtrimfile):
    embed = np.load(embedtrimfile).astype(np.float32)
    return embed


def map_data2id(data_example, vocabfile):
    vocabid = _load_vocab(vocabfile)
    sentence = data_example.sentence
    for i, w in enumerate(sentence):
        m = vocabid[w]
        data_example.sentence[i] = m
    sen_len = len(data_example.sentence)
    if sen_len < FLAGS.max_len:
        data_example.sentence.extend([0] * (FLAGS.max_len - sen_len))

def _entity_feature(data_example):
    """
    文章中将WF特征表示为[x_0],[x_1],....
    但实际上仅仅以该单词的词向量作为WF特征就已经很合适，所以下面会给出两种WF的方案。
    :return:
    """
    entity1_index = data_example.entity1.first
    entity2_index = data_example.entity2.first
    entity1 =data_example.sentence[entity1_index]
    entity2 =data_example.sentence[entity2_index]
    return entity1,entity2


def _position_feature(data_example):
    # 位置特征是为了充分考虑entity特征，此外也可以考虑依存树特征。
    def _get_position(n):
        if n < -60:
            return 0
        elif n >= -60 and n <= 60:
            return n + 61
        return 122
    entity_1_first = data_example.entity1.first
    entity_2_first = data_example.entity2.first
    position1 = []
    position2 = []
    length = len(data_example.sentence)
    for i in range(length):
        position1.append(_get_position(i - entity_1_first))
        position2.append(_get_position(i - entity_2_first))
    # position = []
    # for i, pos in enumerate(position2):
    #     position.append([ position1[i],pos])
    return position1,position2

def build_sequence_example(data_example):
    """
    用tf.train.SequenceExample()函数将之前所做的特征存储起来。
    context 来放置非序列化部分；如：lexical，label(对于一个实例而言label是一个非序列化的数据)
    feature_lists 放置变长序列。如：WF,PF
    :param data_example:
    :return:example
    """
    position1,position2 = _position_feature(data_example)
    entity1,entity2 = _entity_feature(data_example)
    example = tf.train.SequenceExample()

    example_label = int(data_example.label)
    example.context.feature["label"].int64_list.value.append(example_label)
    example.context.feature["entity1"].int64_list.value.append(entity1)
    example.context.feature["entity2"].int64_list.value.append(entity2)
    sentence = data_example.sentence
    for w in sentence:
        example.feature_lists.feature_list["sentence"].feature.add().int64_list.value.append(w)
    for p in position1:
        example.feature_lists.feature_list["position1"].feature.add().int64_list.value.append(p)
    for k in position2:
        example.feature_lists.feature_list["position2"].feature.add().int64_list.value.append(k)
    return example

def tfrecord_write(data, tfrecordfilename):
    """
    将最初的data数据实例化成data_example数据再写入内存。
    :param data:
    :param filename:
    :return:
    """
    if not os.path.exists(tfrecordfilename):
        with tf.python_io.TFRecordWriter(tfrecordfilename) as writer:
            for data_example in data:
                map_data2id(data_example, FLAGS.vocabfile)
                example = build_sequence_example(data_example)
                writer.write(example.SerializeToString())

def parse_tfrecord(sereialized_example):
    context_features = {
        "label": tf.FixedLenFeature([], tf.int64),
        "entity1": tf.FixedLenFeature([], tf.int64),
        "entity2": tf.FixedLenFeature([], tf.int64),
    }
    sequence_features = {
        "sentence": tf.FixedLenSequenceFeature([], tf.int64),
        "position1": tf.FixedLenSequenceFeature([], tf.int64),
        "position2": tf.FixedLenSequenceFeature([], tf.int64),
    }

    contex_dict, sequence_dic = tf.parse_single_sequence_example(sereialized_example,
                                        context_features=context_features,
                                        sequence_features=sequence_features)
    sentence = sequence_dic["sentence"]
    position1 = sequence_dic["position1"]
    position2 = sequence_dic["position2"]
    label = contex_dict["label"]
    entity1 = contex_dict["entity1"]
    entity2 = contex_dict["entity2"]
    return label,position1,position2, sentence,entity1,entity2

def read_data_as_batch(tfrecordfilename, epoch, batchsize, shuffle=True):
    serized_data = tf.data.TFRecordDataset(tfrecordfilename)
    serized_data = serized_data.map(parse_tfrecord)
    serized_data = serized_data.repeat(epoch)
    if shuffle:
        serized_data = serized_data.shuffle(buffer_size=500)
    serized_data = serized_data.batch(batchsize)
    iterator = serized_data.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch

def inputs():
    train_data = load_raw_data(FLAGS.train_file)
    test_data = load_raw_data(FLAGS.test_file)
    vocab2id(train_data, test_data, FLAGS.vocabfile)
    embed_trim(FLAGS.vocabfile, FLAGS.embedtrimfile)
    embed = _load_embed(FLAGS.embedtrimfile)
    tfrecord_write(train_data, FLAGS.tfrecordfilename_train)
    tfrecord_write(test_data, FLAGS.tfrecordfilename_test)
    train = read_data_as_batch(FLAGS.tfrecordfilename_train, FLAGS.epoch, FLAGS.batchsize)
    test = read_data_as_batch(FLAGS.tfrecordfilename_test, FLAGS.epoch, batchsize=2717, shuffle=False)
    return train, test, embed

def write(label2idpath, test_resultpath, rediction_label):
    with open(label2idpath, 'rb') as rf:
        with open(test_resultpath, 'w') as wf:
            label2id_dir = pickle.load(rf)
            id2labeldir = {i: label for label, i in label2id_dir.items()}
            for i, relation in enumerate(rediction_label):
                wf.write("{}\t{}\n".format(i + 8001, id2labeldir[relation]))


if __name__ == "__main__":
    """
    测试：
    """
    flags = tf.app.flags
    flags.DEFINE_string("train_file", "../data/train", "train_file")
    flags.DEFINE_string("test_file", "../data/test", "test_file")
    flags.DEFINE_string("vocabfile", "../data/vocab/vocabfile.pkl", "vocabfile")
    flags.DEFINE_string("embedtrimfile", "../data/embedtrimfile.npy", "embedtrimfile")
    flags.DEFINE_string("position_init_embed", "../data/position_init_embed.npy", "position_init_embed")
    flags.DEFINE_string("tfrecordfilename_train", "../data/tfrecord/tfrecordfilename_train", "tfrecordfilename_train")
    flags.DEFINE_string("tfrecordfilename_test", "../data/tfrecord/tfrecordfilename_test", "tfrecordfilename_test")
    flags.DEFINE_integer("epoch", 10, "epoch")
    flags.DEFINE_integer("embed_num", 123, "embed_num")
    flags.DEFINE_integer("num_classes", 10, "max_len")
    flags.DEFINE_integer("max_len", 10, "max_len")
    flags.DEFINE_integer("batchsize", 10, "batchsize")
    with tf.Session() as sess:
        train, test, embed = inputs()
        try:
            while True:
                print(sess.run(train))

        except tf.errors.OutOfRangeError:
            print("end!")




