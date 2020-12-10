#!usr/bin/env python3
# -*- coding: utf-8 -*-
'a K Nearest Neighbor classifier'
__author__ = 'Ronaldo Wang'

import numpy as np
import re, random
from functools import reduce
from math import sqrt

test_num = 150       #测试集邮件数量
times = 20          #循环次数
k_num = 10

#分词函数
def word_seg(text):
    listofwords = re.split(r'\W+', text)
    return [word.lower() for word in listofwords if len(word) > 1]      #全部变为小写，以便缩小最终词汇表

#将多个邮件的词汇表整合为一个总体词汇表，并除去停用词
def Vocabcreate(doclist):
    vocab_set = set([])
    for document in doclist:
        vocab_set = vocab_set | set(document)
    stop_set = set(['I', 'the', 'an', 'a', 'that', 'this', 'those', 'these', 'you', 'me', 'it', 'she', 'he', 'spam', 'ham'])   #设置停用词
    vocab_set = vocab_set - stop_set
    return list(vocab_set)

#由每封邮件的词汇表构建其文档向量，向量维度为总体词汇表长度
def Word2Vec(vocabulary, doc_words):
    Vec = list(np.zeros(len(vocabulary), dtype = np.int))
    for word in doc_words:
        if word in vocabulary:
            Vec[vocabulary.index(word)] = 1
    return Vec

def classify(word_vector, train_class, train_matrix, k):
    similarity = []
    for i in range(len(train_matrix)):
        mul = sum(word_vector * train_matrix[i])
        sqr_test = [x * x for x in word_vector]
        sqr_train = [x * x for x in train_matrix[i]]
        square = (sum(sqr_test) * sum(sqr_train))
        #print(square)
        sim = float(mul) / sqrt(square)
        tmp_vec = (sim, train_class[i])
        #print(dis, train_class[i])
        similarity.append(tmp_vec)
    similarity = sorted(similarity, reverse = True)
    t = 0
    for i in range(k):
        t += similarity[i][1]
    if float(t) / k > 0.5:
        return 1
    else:
        return 0

def Mail_classify():
    doc_list = []
    class_list = []                 #文档类别向量，1为spam，0为ham
    #从邮件语料库获取所有邮件
    for i in range(1, 292):
        ham_mail = open('C:\\Users\\Wanghanhui\\Desktop\\AI论文程序及数据\\数据集\\email\\ham\\%d.txt' % i, 'r').read()
        tmp_word_list = word_seg(ham_mail)      #分词步骤，为每一封邮件划分词汇表
        doc_list.append(tmp_word_list)          #doc_list最终组织结构为：[[ham1词汇表], ..., [spam1词汇表], ..., [spam358词汇表]]
        class_list.append(0)                    #ham对应标签为0
    for i in range(1, 358):
        spam_mail = open('C:\\Users\\Wanghanhui\\Desktop\\AI论文程序及数据\\数据集\\email\\spam\\%d.txt' % i, 'r').read()
        tmp_word_list = word_seg(spam_mail)
        doc_list.append(tmp_word_list)
        class_list.append(1)                    #spam对应标签为1
    vocab_list = Vocabcreate(doc_list)          #为所有的邮件中的词汇做一个整体词汇表(即将doc_list中各个内层list合并去重)
    #从650封邮件中划分出test_num封作为测试集
    train_set = list(range(len(class_list)))
    test_set = []
    for i in range(test_num):
        ran_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[ran_index])
        del (train_set[ran_index])
    train_matrix = []
    train_class = []
    for doc_index in train_set:
        train_class.append(class_list[doc_index])           
        train_matrix.append(Word2Vec(vocab_list, doc_list[doc_index]))          #由文档词汇表和总体词汇表生成文档向量并组装为文档矩阵
    Train_class, Train_matrix = np.array(train_class), np.array(train_matrix)
    test_error = 0
    correct = [0, 0]
    error = [0, 0]
    i = 0
    for doc_index in test_set:
        if i > 99:
            break
        i += 1
        word_vec = np.array(Word2Vec(vocab_list, doc_list[doc_index]))      #为测试集邮件构建文档向量
        test_type = classify(word_vec, Train_class, Train_matrix, k_num)            #利用k-NN分类器进行分类
        if test_type != class_list[doc_index]:          #说明分类错误
            test_error += 1
            #print('分类错误测试集：', doc_list[doc_index])
            if test_type == 1:              #判断分类错误类型
                #print('将ham错分为spam 权重大')
                error[1] += 1
            else:
                #print('将spam错分为ham 权重小')
                error[0] += 1
        else:
            if test_type == 1:
                correct[1] += 1
            else:
                correct[0] += 1
    #print('Error rate: %.2f%%' % (float(test_error) / len(test_set) * 100))
    #print('正确分类 ham: %d spam: %d' % (correct[0], correct[1]))
    return correct[1], error[1], error[0], correct[0]

if __name__ == '__main__':
    print('A--将spam判断为spam, B--将ham判断为spam, C--将spam判断为ham, D--将ham判断为ham')
    A, B, C, D = 0, 0, 0, 0
    rate = 100 / 100.0
    for i in range(times):
        dA, dB, dC, dD = Mail_classify()
        A += dA
        B += dB
        C += dC
        D += dD
    print('A: %d, B: %d, C: %d, D: %d' % (A, B, C, D))
    print('A: %.2f%%, B: %.2f%%, C: %.2f%%, D: %.2f%%' % (A / (rate * times), B / (rate * times), C / (rate * times), D / (rate * times)))
    #Mail_classify()