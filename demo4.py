'''
这个是对demo1的修改，主要针对的问题是将每个种群的进化独立进行，为每个种群提供一个最大值，让每个种群自己去进化，而不会收到全局最大值的影响。
在demo1中每次梯度的更新都会正对全局最大值。
'''
import tensorflow as tf
import multiprocessing
import threading
import math
import random
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from minepy import MINE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn import neighbors
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import pymrmr

'''
首先就是创建全局网络
'''


class Global_pop(object):
    def __init__(self, name, data):
        with tf.variable_scope(name):
            self.name = name
            self.DNA_size = data.DNA_size
            self.max_fit = 0.0
            self.dr = 0.0
            self.pop_list = []
            self.fit_val = 100.0
            self.workPop = tf.Variable(0)
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.05, mean=0.5), dtype=tf.float32,
                                        name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.DNA_size), dtype=tf.float32, name=name + '_cov')
            self.mvn = MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov))
            self.mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/mean')
            self.cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/cov')

    def getMaxfit(self):
        return self.max_fit

    def getDr(self):
        return self.dr

    def setMaxfit(self, fit):
        self.max_fit = fit

    def setDr(self, dr):
        self.dr = dr

    def setFit_val(self, fit, name, g):
        lock_global1.acquire()
        if g == 0:
            val = []
            val.append(name)
            val.append(fit)
            self.pop_list.append(val)
        else:
            for i in self.pop_list:
                if name == i[0]:
                    i[1] = fit
        lock_global1.release()

    def is_choose(self, name):
        i_name = ''
        i_fit = 100.0
        is_choose = False
        lock_global2.acquire()
        for i in self.pop_list:
            if i[1] < i_fit:
                i_name = i[0]
                i_fit = i[1]
        lock_global2.release()
        if i_name == name:
            is_choose = True
        return is_choose

    def getPop_list(self):
        return self.pop_list
'''
然后就是创建每个子线程的网络
'''


class Worker_pop(object):
    def __init__(self, name, data, global_pop, bili):
        with tf.variable_scope(name):
            self.name = name
            self.max_fit = 0.0
            self.fit_val = 0.0
            self.dr = 0.0
            self.bili = bili
            self.DNA_size = data.getDNA_size()
            # self.mean_params, self.cov_params, self.mean, self.cov = self._creat_net(name, data.getDNA_size())
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.05, mean=0.5), dtype=tf.float32,
                                        name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.DNA_size), dtype=tf.float32, name=name + '_cov')
            self.mvn = MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov))
            self.make_kid = self.mvn.sample(N_POP)
            self.tfkids_fit = tf.placeholder(tf.float32, [math.floor(N_POP/Factor), ])
            self.tfkids = tf.placeholder(tf.float32, [math.floor(N_POP/Factor), self.DNA_size])
            # self.loss = -tf.reduce_mean(
            #     self.mvn.log_prob(self.tfkids) * self.tfkids_fit + 0.01 * self.mvn.log_prob(
            #         self.tfkids) * self.mvn.prob(
            #         self.tfkids))
            # self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * 0.04 * (self.tfkids_fit ** 3))
            self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * self.tfkids_fit)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            # self.train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
            #     self.loss)  # compute and apply gradients for mean and cov
            self.mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/mean')
            self.cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/cov')
            with tf.name_scope('pull'):
                self.pull_mean_op = self.mean.assign(global_pop.mean)
                self.pull_cov_op = self.cov.assign(global_pop.cov)
                # self.pull_mean_params_op = [l_p.assign(g_p) for l_p, g_p in
                #                             zip(self.mean_params, global_pop.mean_params)]
                # self.pull_cov_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.cov_params, global_pop.cov_params)]
            with tf.name_scope('push'):
                self.push_mean_op = global_pop.mean.assign(self.mean)
                self.push_cov_op = global_pop.cov.assign(self.cov)
                # self.push_mean_params_op = [g_p.assign(l_p) for g_p, l_p in
                #                             zip(global_pop.mean_params, self.mean_params)]
                # self.push_cov_params_op = [g_p.assign(l_p) for g_p, l_p in zip(global_pop.cov_params, self.cov_params)]
            with tf.name_scope('restart'):
                self.re_mean_op = self.mean.assign(tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.05, mean=0.5), dtype=tf.float32))
                self.re_cov_op = self.cov.assign(tf.Variable(1.0 * tf.eye(self.DNA_size), dtype=tf.float32))

    def _update_net(self):
        sess.run([self.push_mean_op, self.push_cov_op])
        # lock_push.acquire()
        # self.push_mean_params_op = [g_p.assign(l_p) for g_p, l_p in zip(global_pop.mean_params, self.mean_params)]
        # self.push_cov_params_op = [g_p.assign(l_p) for g_p, l_p in zip(global_pop.cov_params, self.cov_params)]
        # sess.run([self.push_mean_params_op, self.push_cov_params_op])
        # self.update_mean = self.train_op.apply_gradients(zip(self.mean_grads, global_pop.mean_params))
        # self.update_cov = self.train_op.apply_gradients(zip(self.cov_grads, global_pop.cov_params))
        # sess.run([self.update_mean, self.update_cov])  # local grads applies to global net
        # lock_push.release()

    def _pull_net(self):
        sess.run([self.pull_mean_op, self.pull_cov_op])
        # lock_pull.acquire()
        # self.pull_mean_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.mean_params, global_pop.mean_params)]
        # self.pull_cov_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.cov_params, global_pop.cov_params)]
        # sess.run([self.pull_mean_params_op, self.pull_cov_params_op])
        # lock_pull.release()

    def _restart_net(self):
        sess.run([self.re_mean_op, self.re_cov_op])

    def getMaxfit(self):
        # self.fit_val = (1 - self.bili) * self.fit_val + self.max_fit
        return self.max_fit

    def setMaxfit(self, fit):
        # self.fit_val = (1 - self.bili) * self.fit_val + fit
        if fit > self.max_fit:
            self.max_fit = fit

    def getFit_val(self):
        return self.fit_val

    def setFit_val(self, fit_list):
        fit_list.sort(reverse=True)
        len_val = math.floor(0.2 * len(fit_list))
        fits_val = 0.0
        for i in range(len_val):
            fits_val = fits_val + 0.2 * fit_list[i] * math.exp(-(i/2))
        # max_pop_fit = 0.0
        # for i in fit_list:
        #     if i > max_pop_fit:
        #         max_pop_fit = i
        self.fit_val = 0.5 * self.fit_val + 0.5 * fits_val

    def getDr(self):
        return self.dr

    def setDr(self, dr):
        self.dr = dr

    def getMvn(self):
        return self.mvn

    def setMvn(self, mvn):
        self.mvn = tf.Variable(mvn.initialized_value())

    def getMean(self):
        return self.mean

    def getCov(self):
        return self.cov

'''
这是一个对数据处理的类
'''
class Dataset(object):
    def __init__(self, filename, type):
        self.file = filename
        self.filename = filename + '.txt'
        self.type = type
        self.bili = 1.0
        self.DNA_size = 0
        self.Feature = []
        self.trainX = []
        self.trainy = []
        self.testX = []
        self.testy = []

    def __loadData__(self, label_loc, div_mode, test_size, select_feature):
        self.label_loc = label_loc
        self.div_mode = div_mode
        self.test_size = test_size
        self.select_feature = select_feature
        self.__pretreatment__()
        if self.label_loc == 0:
            data = pd.read_table(self.filenamed, sep=' ')
            self.fea, self.lab = data.ix[:, 1:], data.ix[:, 0]
            # self.fea, self.lab = shuffle(self.fea, self.lab)
            self.DNA = len(np.array(self.fea)[0])
            if self.DNA < self.select_feature:
                self.select_feature = self.DNA
            elif self.DNA >= self.select_feature:
                self.bili = self.select_feature / self.DNA
        elif self.label_loc == 1:
            data = pd.read_table(self.filenamed, sep=' ')
            self.fea, self.lab = data.ix[:, 0:len(open(self.filename).readline().split(self.div_mode)) - 1], data.ix[:, -1]
            # self.fea, self.lab = shuffle(self.fea, self.lab)
            self.DNA = len(np.array(self.fea)[0])
            if self.DNA < self.select_feature:
                self.select_feature = self.DNA
            elif self.DNA >= self.select_feature:
                self.bili = self.select_feature / self.DNA

    def __pretreatment__(self):
        numData = len(open(self.filename).readline().split(self.div_mode))
        v = []
        val = []
        for i in range(numData):
            v.append(i)
        val.append(v)
        fr = open(self.filename)
        for line in fr.readlines():
            xi = []
            curline = line.strip().split(self.div_mode)
            for i in range(numData):
                xi.append((curline[i]))
            val.append(xi)
            self.filenamed = self.file + 'ed' + '.txt'
            self.saveData(self.filenamed, np.array(val))

    def saveData(self, filename, dataname):
        with open(filename, 'w') as file_object:  # 将文件及其内容存储到变量file_object

            # 写入第一行(第一块)
            file_object.write(str(dataname[0, 0]))  # 写第一行第一列
            for j in range(1, np.size(dataname, 1)):
                file_object.write(' ' + str(dataname[0, j]))  # 写第一列后面的列

            # 写入第一行后面的行（第二块）
            for i in range(1, np.size(dataname, 0)):
                file_object.write('\n' + str(dataname[i, 0]))
                for j in range(1, np.size(dataname, 1)):
                    file_object.write(' ' + str(dataname[i, j]))

    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)

    def __getData__(self):
        # self.train_selected = SelectKBest(lambda X, Y: np.array(map(lambda x: self.mic(x, Y), X.T)).T, k=self.select_feature).fit_transform(self.fea, self.lab)
        # self.train_selected = SelectKBest(chi2, k=self.select_feature).fit_transform(self.fea, self.lab)
        self.train_selected = SelectFromModel(GradientBoostingClassifier()).fit_transform(self.fea, self.lab)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_selected, self.lab, test_size=self.test_size)
        if self.type == 'split':
            self.getMatDataBysplit(self.x_train, self.x_test, self.y_train, self.y_test)
        elif self.type == 'nfold':
            skf = StratifiedKFold(n_splits=self.test_size, shuffle=True)
            skf.get_n_splits(self.train_selected, self.lab)
            self.getMatDataBynfold(skf, self.train_selected, self.lab)
            # print(self.trainX[0])
            # print(self.trainX[0].shape)
            # print(self.trainy[0])
            # print(self.trainy[0].shape)
            # print(self.testy[0])
            # print(self.DNA)
            # print(self.DNA_size)
            # print(self.Feature)


    # 将列表形式的数据转换为矩阵形式
    def getMatDataBysplit(self, x_train, x_test, y_train, y_test):
        self.trainX.append(np.array(x_train))
        self.testX.append(np.array(x_test))
        self.trainy.append(np.array(y_train))
        self.testy.append(np.array(y_test))
        self.DNA_size = len(np.array(x_train)[0])

        for i in range(self.DNA_size):
            self.Feature.append(i)  # 其中的元素是特征所在的位置

    def getMatDataBynfold(self, skf, feature, label):
        for train_index, test_index in skf.split(feature, label):
            print(test_index)
            trainX_ = feature[train_index]
            trainy_ = label[train_index]
            testX_ = feature[test_index]
            testy_ = label[test_index]
            self.trainX.append(trainX_)
            self.testX.append(testX_)
            self.trainy.append(trainy_)
            self.testy.append(testy_)
        self.DNA_size = len(np.array(feature)[0])
        for i in range(self.DNA_size):
            self.Feature.append(i)

    def getDNA_size(self):
        return self.DNA_size

    def getdata(self):
        return self.trainX, self.testX, self.trainy, self.testy

    def getFeature(self):
        return self.Feature

    def getReadType(self):
        return self.type

    def getTestsize(self):
        return self.test_size

    def getbili(self):
        return self.bili

'''
然后就是对每个线程工作内容的定义
'''

class Worker(object):
    def __init__(self, name, data, numfea, classifier, global_pop, factor):
        self.name = name
        self.data = data
        self.numfea = numfea
        self.classifier = classifier
        self.pos = 0
        self.factor = factor
        self.popnet = Worker_pop(name, data, global_pop, 0.5)  # 每个线程在初始化中首先区初始化一个网络

    def numtofea(self, num, fea_list):
        feature = []
        for i in range(len(num)):
            if num[i] == '1':
                feature.append(fea_list[i])
            else:
                continue
        return feature  # 返回的是所选特征所在的位置

    def read_data_fea(self, fea_list, dataset):
        dataMat = np.mat(np.array(dataset))
        col = dataMat.shape[0]  # 行号
        data_sample = []
        for i in range(col):
            col_i = []
            for j in fea_list:
                col_i.append(dataMat[i, j])
            data_sample.append(col_i)
        return data_sample

    def get_fitness(self, data_train, label_train, data_pre, label_pre, train_cla):
        acc = 0.00
        if train_cla is 'train_knn':
            clf = neighbors.KNeighborsClassifier(n_neighbors=1)  # 创建分类器对象
            if len(data_train[0]) > 0:
                clf.fit(data_train, label_train)  # 用训练数据拟合分类器模型搜索
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, np.array(label_pre))
        elif train_cla is 'svm':
            clf = svm.SVC()
            if len(data_train[0]) > 0:
                clf.fit(data_train, label_train)
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, np.array(label_pre))
        elif train_cla is 'tree':
            if len(data_train[0]) > 0:
                clf = DecisionTreeClassifier()
                clf.fit(data_train, label_train)
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, np.array(label_pre))
        return acc * 100

    def get_max(self, new_list):
        max = -101.0
        count = 0
        index = 0
        for i in new_list:
            count = count + 1
            if (i > max):
                max = i
                index = count
        return max, index - 1  # 返回最大的适应的以及所对应的子代

    def acc_pre(self, predict, label_train):
        num = 0
        for i in range(len(predict)):
            if predict[i] != label_train[i]:
                num += 1
        return (1 - (num / len(label_train)))

    def dr_pre(self, feature_list):
        feature_sum = len(feature_list)
        # num = data.getDNA_size()
        count = 0
        for i in feature_list:
            if (i == '1'):
                count = count + 1
        return 1 - (count / self.numfea)

    def insert_value(self, new_max, new_dr, bili):
        lock_max_fit.acquire()
        max_fit = global_pop.getMaxfit()
        lock_max_fit.release()
        lock_dr.acquire()
        dr = global_pop.getDr()
        lock_dr.release()
        if (max_fit * bili + dr * 100 * (1 - bili)) < (new_max * bili + new_dr * 100 * (1 - bili)):
            lock_max_fit.acquire()
            global_pop.setMaxfit(new_max)
            lock_max_fit.release()
            lock_dr.acquire()
            global_pop.setDr(new_dr)
            lock_dr.release()
            return True
        else:
            return False

    def getKids_fit(self, kids_fits, kids, factor):
        # kids_fit = []
        # for i in kids_fits:
        #     kids_fit.append(1.2 * (i - max))
        #     # kids_fit.append(i - max)
        # return kids_fit
        # kids_fits.sort(reverse=True)
        val = []
        for i in kids_fits:
            val.append(i)
        kids_fit = []
        kid = []
        for i in range(math.floor(len(val) / factor)):
            ind = 0
            pos = 0.0
            for j in range(len(val)):
                if val[j] > pos:
                    pos = val[j]
                    ind = j
            val[ind] = -1.0
            kid.append(kids[ind])
            kids_fit.append(pos)
        return kids_fit, kid

    def save_to_file(self, file_name, contents):
        fh = open(file_name, 'w')
        fh.write(contents)
        fh.close()

    def save_to_afile(self, contents):
        fh = open('E:/dataset/result.txt', 'a+')
        fh.write(contents)
        fh.close()

    def work(self):
        for g in range(MAX_GLOBAL_EP):
            print(self.name, g+1, '次迭代')
            # if g % 10 == 0 and g != 0:
            #     lock_max_fit.acquire()
            #     max_fit = global_pop.getMaxfit()
            #     lock_max_fit.release()
            #     print(self.name, g + 1, '当前最大值：', max_fit)
                # self.LR = self.LR * pow(0.99, g / 20)
            # print(self.name, g+1, 'mean', sess.run(self.popnet.mvn.mean()))
            # print(self.name, g+1, 'cov', sess.run(self.popnet.mvn.stddev()))
            # print(self.name, g + 1)
            # print(sess.run(self.popnet.mvn.mean()))
            # print(sess.run(self.popnet.mvn.stddev()))
            # if g % 20 == 0 and g != 0:
            #     self.LR = self.LR * pow(0.99, g / 20)
            lock_kids.acquire()
            kids = sess.run(self.popnet.make_kid)
            lock_kids.release()
            kids_fit = []  # 初始化子代的适应度值，这是一个列表，初始化为空
            feature_set = []  # 所有子代选取特征的集合
            kids_fits = []
            for kid in kids:  # 遍历每一个子代
                feature_list = []
                for feature in kid:  # 遍历每个子代的每个特征
                    # if feature > 0.50 * self.data.getbili():  # 对于特征值大于0.5的特征就用1表示
                    # if np.round(np.random.normal(feature, 0.01, 1), 2) > 0.50:
                    if feature > 0.50:
                        feature_list.append('1')
                    else:
                        feature_list.append('0')  # 否则就用0来表示
                feature_Select = self.numtofea(feature_list, self.data.getFeature())  # 将0，1串映射到特征的选取中
                feature_set.append(feature_list)  # 将每个子代所选取的特征放到特征集合中
                if self.data.getReadType() is 'split':
                    trainX_, testX_, trainy_, testy_ = self.data.getdata()
                    trainX = trainX_[0]
                    testX = testX_[0]
                    trainy = trainy_[0]
                    testy = testy_[0]
                    data_sample = self.read_data_fea(feature_Select, trainX)
                    data_predict = self.read_data_fea(feature_Select, testX)  # 找到所选特征所对应的数据集
                    kid_fit = self.get_fitness(data_sample, trainy, data_predict, testy,
                                               self.classifier)  # 然后更具所选的特征集合进行准确率求解
                    lock_max_fit.acquire()
                    self.popnet.setMaxfit(kid_fit)
                    lock_max_fit.release()
                    # kid_tifs = kid_fit - max
                    # kids_fit.append(kid_tifs)
                    kids_fits.append(kid_fit)  # 然后将每一个子代的适应度添加到适应度的集合当中
                elif self.data.getReadType() is 'nfold':
                    kid_fit = 0.0
                    for i in range(self.data.getTestsize()):
                        trainX_, testX_, trainy_, testy_ = self.data.getdata()
                        trainX = trainX_[i]
                        testX = testX_[i]
                        trainy = trainy_[i]
                        testy = testy_[i]
                        data_sample = self.read_data_fea(feature_Select, trainX)
                        data_predict = self.read_data_fea(feature_Select, testX)  # 找到所选特征所对应的数据集
                        kid_fit_sample = self.get_fitness(data_sample, trainy, data_predict, testy,
                                                   self.classifier)  # 然后更具所选的特征集合进行准确率求解print(kid_fit_sample)
                        kid_fit = kid_fit + kid_fit_sample
                    kid_fit = kid_fit / self.data.getTestsize()
                    # lock_max_fit.acquire()
                    # self.popnet.setMaxfit(kid_fit)
                    # lock_max_fit.release()
                    # kid_tifs = kid_fit - max
                    # kids_fit.append(kid_tifs)
                    kids_fits.append(kid_fit)  # 然后将每一个子代的适应度添加到适应度的集合当中
            # lock_max_fit.acquire()
            # max = self.popnet.getMaxfit()
            # lock_max_fit.release()
            self.popnet.setFit_val(kids_fits)
            lock_Fit_val.acquire()
            global_pop.setFit_val(self.popnet.getFit_val(), self.name, g)
            lock_Fit_val.release()
            kids_fit, kid = self.getKids_fit(kids_fits, kids, self.factor)
            # kids_fit = self.getKids_fit(kids_fits, max)
            sess.run(self.popnet.train_op,
                     {self.popnet.tfkids_fit: kids_fit, self.popnet.tfkids: kid})  # 然后根据所有子代的适应度来进行参数更新
            new_max, count = self.get_max(kids_fits)  # 找到子代中准确率最高的孩子，以及返回孩子的位置
            feature_get = feature_set[count]  # 根据找到的位置找到相应的特征选取
            new_dr = self.dr_pre(feature_get)  # 然后更具特征选取来求出所对应的维度缩减率
            # if new_max > self.max_fit:
            #     self.max_fit = new_max
            # str1 = '%s在第%d次时的fit%f' % (self.name, g, self.popnet.getFit_val()) + '\n'
            # self.save_to_afile(str1)
            changed = self.insert_value(new_max, new_dr, 1)  # 将新得到的值一目前最大值进行比较
            if changed is True:
                info1 = '   %s使用knn第%d轮迭代' % (self.name, (g+1)) + '\n'
                info2 = '选取特征为： ' + ' '.join(feature_get) + '\n'
                info3 = '维度缩减为： %f' % new_dr + '\n'
                info4 = '准确率为： %f' % new_max + '\n'
                info = info1 + info2 + info3 + info4
                self.save_to_afile(info)
                print(info)
                # print('   ', self.name, '使用', 'knn', '第', g + 1, '轮迭代：')
                # print('选取特征为：', feature_get)
                # print('维度缩减为：', new_dr)
                # print('准确率为：', new_max)s
                print(kids_fits)
                self.popnet._update_net()
                info5 = ' '.join(str(i) for i in sess.run(self.popnet.mvn.mean())) + '\n'
                info6 = '%s在第%d次迭代时更新全局网络为：' % (self.name, g+1)
                self.save_to_afile(info6)
                self.save_to_afile(info5)
                print(self.name, g + 1, '更新全局网络为：', sess.run(self.popnet.mvn.mean()))
            #     print(self.name, g+1, '更新全局网络为：', sess.run(self.popnet.mvn.mean()))
            val = 1 / (1 + math.exp(-(((g-self.pos) / 100) - 3.5)))
            lock_is_choose.acquire()
            is_choose = global_pop.is_choose(self.name)
            lock_is_choose.release()
            # if is_choose:
            if random.random() < val and is_choose:
                self.pos = g
            #     print(self.name, g+1, '当前网络为:', sess.run(self.popnet.mvn.mean()))
                if random.random() < 0.3:
                    self.popnet._pull_net()
                    info7 = ' '.join(str(i) for i in sess.run(self.popnet.mvn.mean())) + '\n'
                    info8 = '%s在第%d次迭代时获取全局网络为：' % (self.name, g+1)
                    self.save_to_afile(info8)
                    self.save_to_afile(info7)
                    print(self.name, g+1, '获取目标网络为：', sess.run(self.popnet.mvn.mean()))
                else:
                    self.popnet._restart_net()
                    info9 = ' '.join(str(i) for i in sess.run(self.popnet.mvn.mean())) + '\n'
                    info0 = '%s在第%d次迭代时重启网络为：' % (self.name, g + 1)
                    self.save_to_afile(info0)
                    self.save_to_afile(info9)
                    print(self.name, g + 1, '重启网络为：', sess.run(self.popnet.mvn.mean()))



if __name__ == '__main__':
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        N_WORKERS = 8    # 表示运行的线程个数
        N_POP = 120    # 表示生成子代的个数

        # 接着就是定义一些锁来对共享资源进行锁定
        lock_kids = threading.Lock()
        lock_max_fit = threading.Lock()
        lock_dr = threading.Lock()
        lock_push = threading.Lock()
        lock_pull = threading.Lock()
        lock_Fit_val = threading.Lock()
        lock_is_choose = threading.Lock()
        lock_global1 = threading.Lock()
        lock_global2 = threading.Lock()

        LEARNING_RATE = 0.001    # 表示算法的学习率
        MAX_GLOBAL_EP = 301    # 表示算法的迭代次数

        # 定义分类器
        KNN = 'train_knn'
        SVM = 'svm'
        TREE = 'tree'

        # 新建sess来运行算法
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options))

        # 首先是将数据集特征的第一步过滤
        Factor = 2
        data = Dataset('E:/dataset/arcene', 'split')
        data.__loadData__(1, ',', 0.3, 500)
        data.__getData__()
        # 然后就是初始化全局网络
        global_pop = Global_pop('global', data)

        fh = open('E:/dataset/result.txt', 'w')
        fh.write('begin:\n')
        fh.close()

        # 接着就是通过核心数来创建多线程进行工作
        with tf.device("/gpu:0"):
            workers = []
            for i in range(N_WORKERS):
                with tf.device('/gpu:%d' % (i+1)):
                    with tf.name_scope('GPU_%d' % (i+1)) as scope:
                        i_name = 'W_%i' % (i+1)  # worker name
                        workers.append(Worker(i_name, data, data.DNA, KNN, global_pop, Factor))

            COORD = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                job = lambda: worker.work()
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)
            COORD.join(worker_threads)