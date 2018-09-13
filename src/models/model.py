#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/8/26
"""
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def input_attention(entity1_embed, entity2_embed, sentences, nposition1, nposition2,is_train=True):
    """
    构建input_attention
    :param entity1: 【batchsize,sentencesize,1】
    :param entity2:
    :param sentence_feature:[batchsize,sentence,3]
    :return:
    """
    def _get_a(entity_embed,sentence,num):
        A_j =tf.expand_dims(entity_embed,axis=-1)
        a_j = tf.matmul(sentence,A_j)
        a_j = tf.reshape(a_j,[-1,FLAGS.max_len])
        a_j = tf.nn.softmax(a_j,axis=1)
        return a_j
    def _get_z(position1,position2,sentence):
        result = tf.concat([sentence,position1,position2],axis=2)
        return result

    z = _get_z(nposition1, nposition2, sentences)
    a_1 = _get_a(entity1_embed,sentences,1)
    a_2 = _get_a(entity2_embed,sentences,2)
    alpha = tf.div(tf.add(a_1,a_2),2)
    alpha = tf.tile(tf.reshape(alpha,[-1,FLAGS.max_len,1]), [1, 1, FLAGS.word_dim+2*FLAGS.position_dim])
    input_ = tf.multiply(z,alpha)
    #这里与论文有明显的差别，当使用input_attention时效果较差（use input_），当不使用(use z)时效果反而比较好。

    return z

def attention_pooling(conv_output,label_embed,num_filter,numclass,):
    U = tf.get_variable("U",[num_filter,numclass])
    middle = tf.matmul(tf.reshape(conv_output,[-1,num_filter]),U)
    pool = tf.matmul(middle,label_embed)
    pool = tf.reshape(pool,[-1,FLAGS.max_len,FLAGS.label_dim])
    n_pool = tf.nn.softmax(pool,axis=1)
    conv = tf.reshape(conv_output,[-1,FLAGS.max_len,num_filter])
    conv = tf.transpose(conv, [0, 2, 1])
    atten_pool = tf.matmul(conv,n_pool)
    max_pool = tf.nn.max_pool(tf.expand_dims(atten_pool,-1), ksize=[1, 1,FLAGS.label_dim, 1], strides=[1,1,FLAGS.label_dim, 1], padding="SAME")
    max_pool = tf.reshape(max_pool,[-1,num_filter])
    return max_pool

def cnn_layer(name, input_data, num_filter,label_embed, numclass,):
    dim = input_data.shape[2]
    with tf.variable_scope(name):
        with tf.variable_scope("conv-1"):
            conv_weight = tf.get_variable("Weight", [3, dim, 1, num_filter],
                                          initializer=tf.truncated_normal_initializer(0.1))
            bias = tf.get_variable("bias", [num_filter], initializer=tf.constant_initializer(0.1))
            input_ = tf.expand_dims(input_data,axis=-1)
            conv = tf.nn.conv2d(input_, conv_weight, strides=[1, 1, dim, 1], padding="SAME")
            R_tanh = tf.nn.tanh(tf.nn.bias_add(conv, bias))#
            #R_tanh = tf.reshape(R_tanh,[-1,FLAGS.max_len,num_filter])
            max_pool = tf.nn.max_pool(R_tanh,ksize=[1, FLAGS.max_len,1, 1], strides=[1,FLAGS.max_len,1, 1], padding="SAME")
            #max_pool = attention_pooling(R_tanh,label_embed,num_filter,numclass)
            max_pool = tf.reshape(max_pool,[-1,num_filter])
            return max_pool#[batchsize,num_filter]

class Model(object):
    def __init__(self, word_embed, data, num_classes,num_filter, position_dim, is_train=True,regulizer = True):
        label, position1, position2, sentence, entity1, entity2 = data
        word_embeds = tf.get_variable("word_embed", initializer=word_embed,dtype=tf.float32, trainable=True)
        possition_embed = tf.get_variable('possition_embed', shape=[2 * FLAGS.embed_num, position_dim],
                                          initializer=tf.truncated_normal_initializer(0.1), trainable=True)
        label_embed = tf.get_variable("label_embed",shape=[num_classes,FLAGS.label_dim],initializer=tf.truncated_normal_initializer(0.1), trainable=True)
        real_label = tf.nn.embedding_lookup(label_embed,label)
        sentences = tf.nn.embedding_lookup(word_embeds, sentence)
        entity1_embed = tf.nn.embedding_lookup(word_embeds, entity1)
        entity2_embed = tf.nn.embedding_lookup(word_embeds, entity2)
        nposition1 = tf.nn.embedding_lookup(possition_embed, position1)
        nposition2 = tf.nn.embedding_lookup(possition_embed, position2)
        input_data = input_attention(entity1_embed, entity2_embed, sentences, nposition1, nposition2,is_train)
        w_0 = cnn_layer("feature", input_data, num_filter,label_embed,num_classes)

        acc,predict = self.predict_label(w_0,label_embed,label)
        loss = self.loss(w_0,label_embed,real_label,label)
        if regulizer:
            tv = tf.trainable_variables()
            regulization_loss = FLAGS.l2_learning*tf.add_n([tf.nn.l2_loss(v) for v in tv])
            loss = loss+regulization_loss
        self.accuracy = acc
        self.loss = loss
        self.label = predict
        if not is_train:
            return
        global_step = tf.Variable(0, trainable=False, name='sep', dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(FLAGS.learningrate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = optimizer.minimize(loss, global_step)
        self.global_step = global_step

    def predict_distance(self,matrix1,matrix2):
        wo= tf.tile(tf.expand_dims(tf.nn.l2_normalize(matrix1,1), axis=1), [1, FLAGS.num_classes, 1])
        all_distance = tf.norm(wo - tf.nn.l2_normalize(matrix2,1), axis=2)
        return all_distance

    def predict_label(self,w_0,label_embed,label):
        distance = self.predict_distance(w_0,label_embed)
        predict = tf.argmin(distance, axis=1)

        result = tf.cast(tf.equal(predict,label), dtype=tf.float32)
        acc = tf.reduce_mean(result)
        return acc,predict

    def loss_distance(self,matrix1,matrix2):
        disance = tf.nn.l2_normalize(matrix1,1)-tf.nn.l2_normalize(matrix2,1)
        distance = tf.norm(disance,axis=1)
        return distance

    def loss(self,w_0, label_embed,real_label,label):
        distance = self.predict_distance(w_0, label_embed)
        mask = tf.one_hot(label,FLAGS.num_classes,on_value=100.,off_value=0.)
        neg_label = tf.argmin(tf.add(distance,mask),axis=1)
        flabel_embed = tf.nn.embedding_lookup(label_embed,neg_label)
        distance1 = self.loss_distance(w_0,real_label)
        distance2 = self.loss_distance(w_0,flabel_embed)
        loss = tf.reduce_mean(distance1+(1-distance2))
        return loss

def train_or_valid(train_data, test_data, word_embed):
    with tf.name_scope("Train"):
        with tf.variable_scope("model", reuse=None):
            mtrain = Model(word_embed, train_data, FLAGS.num_classes,FLAGS.num_filter, FLAGS.position_dim, is_train=True,regulizer = True)

    with tf.name_scope("test"):
        with tf.variable_scope("model", reuse=True):
            mtest = Model(word_embed, test_data, FLAGS.num_classes,FLAGS.num_filter, FLAGS.position_dim, is_train=False,regulizer = True)
    return mtrain, mtest

class set_save(object):
    @classmethod
    def save_model(cls, sess, model_path):
        cls.saver = tf.train.Saver()
        cls.saver.save(sess, model_path)

    @classmethod
    def load_model(cls, sess, model_path):
        cls.saver = tf.train.Saver()
        cls.saver.restore(sess, model_path)

