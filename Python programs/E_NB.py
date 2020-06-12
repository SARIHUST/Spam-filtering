#!usr/bin/env python3
# -*- coding: utf-8 -*-
'a Naive Bayes classifier'
__author__ = 'Ronaldo Wang'

import numpy as np
import re, random

test_num = 25       #测试集邮件数量
times = 50          #循环次数

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

def trainNB0(class_vec, matrix):
    count_mail = len(matrix)
    count_vocab = len(matrix[0])
    Pspam = sum(class_vec) / float(count_mail)          #sum(class_vec)可获取训练集中spam数量
    p0array = np.ones(count_vocab)
    p1array = np.ones(count_vocab)
    n_theta_0 = 2.0           #实现拉普拉斯平滑变换
    n_theta_1 = 2.0
    for i in range(count_mail):
        if class_vec[i] == 1:
            p1array += matrix[i]            #最终获得每个词在spam类中出现的次数+1(相当于实现了拉普拉斯平滑变换)
            n_theta_1 += sum(matrix[i])       #最终获得所有词汇在spam类中出现次数总和
        else:
            p0array += matrix[i]
            n_theta_0 += sum(matrix[i])
    p1Vect = np.log(p1array / n_theta_1)      #取对数将后续的Π求积操作换位Σ求和，防止下溢出
    p0Vect = np.log(p0array / n_theta_0)
    return p0Vect, p1Vect, Pspam


def classify(word_vector, p0_vec, p1_vec, pSpam):
    p1 = sum(word_vector * p1_vec) + np.log(pSpam)              #计算属于spam的“概率”
    p0 = sum(word_vector * p0_vec) + np.log(1.0 - pSpam)        #计算属于ham的“概率”
    if p1 > p0:
        return 1
    else:
        return 0

def Mail_classify():
    doc_list = []
    class_list = []                 #文档类别向量，1为spam，0为ham
    #从邮件语料库获取所有邮件
    for i in range(1, 292):
        ham_mail = open('C:\\Users\\Wanghanhui\\Desktop\\AI论文程序及数据\\数据集\\email\\ham\\\\%d.txt' % i, 'r').read()
        tmp_word_list = word_seg(ham_mail)      #分词步骤，为每一封邮件划分词汇表
        doc_list.append(tmp_word_list)          #doc_list最终组织结构为：[[ham1词汇表], [spam1词汇表], ..., [spam25词汇表]]
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
    #最后test_set中的元素为test_num个从0至659的数字，余下在train_set中
    #下面构造训练集的文档矩阵，并利用朴素贝叶斯分类器进行训练
    train_matrix = []
    train_class = []
    for doc_index in train_set:
        train_class.append(class_list[doc_index])           
        train_matrix.append(Word2Vec(vocab_list, doc_list[doc_index]))          #由文档词汇表和总体词汇表生成文档向量并组装为文档矩阵
    Train_class, Train_matrix = np.array(train_class), np.array(train_matrix)
    p0V, p1V, p_spam = trainNB0(Train_class, Train_matrix)             #朴素贝叶斯
    #下面对测试集进行分类检验
    test_error = 0
    correct = [0, 0]
    error = [0, 0]
    for doc_index in test_set:
        word_vec = np.array(Word2Vec(vocab_list, doc_list[doc_index]))      #为测试集邮件构建文档向量
        test_type = classify(word_vec, p0V, p1V, p_spam)            #利用贝叶斯分类器进行分类
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
    rate = test_num / 100.0
    for i in range(times):
        dA, dB, dC, dD = Mail_classify()
        A += dA
        B += dB
        C += dC
        D += dD
    print('A: %d, B: %d, C: %d, D: %d' % (A, B, C, D))
    print('A: %.2f%%, B: %.2f%%, C: %.2f%%, D: %.2f%%' % (A / (rate * times), B / (rate * times), C / (rate * times), D / (rate * times)))
    #Mail_classify()



'''
分词
Input:邮件的txt文件Email
    将Email按照特殊字符和空白符处中断规则分词
    获得Email的词汇列表E_vocab
Output:邮件的词汇列表E_vocab

语料库构建
Input:所有邮件的E_vocab组成的vocabs集合
    创建return_vocab集合
    for E_vocab in vocabs:
        将E_vocab和return_vocab取并集
    创建停用词集合stop_vocab
    all_voab = return_vocab - stop_vocab
Output:all_vocab

生成文档向量
Input:语料库all_vocab, 每封邮件的E_vocab
    创建n维零向量Vec，n为count(all_vocab)
    for word in all_vocab:
        if word在E_vocab中出现
            将word在Vec中对应的元素置为1
Output:Vec

训练朴素贝叶斯分类器
Input:Train_matrix, Train_class
    获取测试集总邮件数count(mail)即Train_matrix行数(Train_class元素数)
    获取总次数count(vocab)即Train_matrix列数
    训练集垃圾邮件占比pSpam = (Train_class元素和) / count(mail)
    创建两个长为count(vocab)的向量p0vec和p1vec，元素全部置为1
    n_theta_0 = n_theta_1 = 2.0
    for i in 0到count(mail) - 1
        if Train_class[i] == 1
            p1vec += Train_matrix[i]
            n_theta_1 += matrix[i]中各元素和，即某一垃圾邮件的词条数
        else
            p0vec += Train_matrix[i]
            n_theta_0 += matrix[i]中各元素和，即某一正常邮件的词条数
    创建向量p0vect，其元素为p0vec中所有元素除以n_theta_0后再取自然对数
    创建向量p1vect，其元素为p1vec中所有元素除以n_theta_1后再取自然对数
Output p0vect, p1vect, pSpam

测试集邮件分类
Input:测试集邮件序列Test_mails, 测试集邮件类别向量Test_class
    correct = error = [0, 0]    将两种分类正确和错误的情形数初始化
    for test_mail in Test_mails
        test_vec = Word2Vec(test_mail)  将测试集邮件向量化
        获得测试邮件分类test_type = classify(test_vec)
        if 测试邮件分类与真实分类不一致
            if 真实分类为Spam
                error[0] += 1
            else
                error[1] += 1
        else
            if 真实分类为Spam
                correct[1] += 1
            else
                correct[0] += 1
Output:correct[1], error[1], error[0], correct[0]

classify
Input:p0vect, p1vect, pSpam, test_vec
    计算文档为Ham的后验概率p0
        测试集文档向量test_vec和p0vect做内积后元素求和再加上1-pSpam取对数
    计算文档为Spam的后验概率p1
        测试集文档向量test_vec和p1vect做内积后元素求和再加上pSpam取对数
    if p1 > p0
        class_type = 1
    else
        class_type = 0
Output:class_type
'''


