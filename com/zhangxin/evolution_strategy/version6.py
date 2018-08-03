'''
这是多线程的第二个版本
主要解决版本一种每个子线程玩咯更新的问题
'''
import tensorflow as tf
import multiprocessing
import threading
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

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
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.02, mean=0.0), dtype=tf.float32,
                                   name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.DNA_size), dtype=tf.float32, name=name + '_cov')
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
'''
然后就是创建每个子线程的网络
'''

class Worker_pop(object):
    def __init__(self, name, data):
        with tf.variable_scope(name):
            self.name = name
            self.DNA_size = data.getDNA_size()
            # self.mean_params, self.cov_params, self.mean, self.cov = self._creat_net(name, data.getDNA_size())
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.1, mean=0.0), dtype=tf.float32,
                                   name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.DNA_size), dtype=tf.float32, name=name + '_cov')
            # self.mvn = MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=self.cov)
            self.mvn = MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov))
            # self.mvn = MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov + tf.Variable(0.05 * tf.eye(self.DNA_size), dtype=tf.float32)))
            self.make_kid = self.mvn.sample(N_POP)
            self.tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
            self.tfkids = tf.placeholder(tf.float32, [N_POP, self.DNA_size])
            self.loss = -tf.reduce_mean(
                self.mvn.log_prob(self.tfkids) * self.tfkids_fit + 0.001 * self.mvn.log_prob(self.tfkids) * self.mvn.prob(
                    self.tfkids))
            self.train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)  # compute and apply gradients for mean and cov
            self.mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/mean')
            self.cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/cov')
        # sess.run(tf.global_variables_initializer())

    # def _creat_net(self, scope, DNA_size):
    #     with tf.variable_scope('mean'):
    #         mean = tf.Variable(tf.truncated_normal([DNA_size, ], stddev=0.02, mean=2.5), dtype=tf.float32,
    #                            name=scope + '_mean')
    #     with tf.variable_scope('cov'):
    #         cov = tf.Variable(1.0 * tf.eye(DNA_size), dtype=tf.float32, name=scope + '_cov')
    #     # with tf.variable_scope('mvn'):
    #     #     mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov +
    #     #         tf.Variable(0.05 * tf.eye(DNA_size), dtype=tf.float32))
    #     mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/mean')
    #     cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/cov')
    #     # sess.run(tf.global_variables_initializer())
    #     return mean_params, cov_params, mean, cov
    #
    # def _make_kids(self):
    #     lock_kids.acquire()
    #     kids = sess.run(self.mvn.sample(N_POP))
    #     lock_kids.release()
    #     return kids

    def _update_net(self):
        lock_push.acquire()
        self.push_mean_params_op = [g_p.assign(l_p) for g_p, l_p in zip(global_pop.mean_params, self.mean_params)]
        self.push_cov_params_op = [g_p.assign(l_p) for g_p, l_p in zip(global_pop.cov_params, self.cov_params)]
        sess.run([self.push_mean_params_op, self.push_cov_params_op])
        # self.update_mean = self.train_op.apply_gradients(zip(self.mean_grads, global_pop.mean_params))
        # self.update_cov = self.train_op.apply_gradients(zip(self.cov_grads, global_pop.cov_params))
        # sess.run([self.update_mean, self.update_cov])  # local grads applies to global net
        lock_push.release()

    def _pull_net(self):
        lock_pull.acquire()
        self.pull_mean_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.mean_params, global_pop.mean_params)]
        self.pull_cov_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.cov_params, global_pop.cov_params)]
        sess.run([self.pull_mean_params_op, self.pull_cov_params_op])
        lock_pull.release()

'''
创建读取数据工具类
首先类中的属性包括有数据维数
'''

class divideData(object):
    def __init__(self):
        self.size = 0
        self.DNA_size = 0
        self.data = []
        self.Feature = []

    def getDNA_size(self):
        return self.DNA_size

    def numtofea(self, num, fea_list):
        feature = []
        for i in range(len(num)):
            if num[i] == 1:
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

    def saveData1(self, filename, dataname):
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

    def dividedata(self, workersize, type):
        for i in range(self.size):
            self.Feature.append(i)  # 其中的元素是特征所在的位置
        for i in range(workersize):
            fea_list = []
            mean = np.random.rand(1, self.DNA_size)
            # mean = tf.Variable(tf.truncated_normal([self.DNA_size, ], stddev=0.1, mean=0.0), dtype=tf.float32)
            if type == 2 or type == 4:
                fea_list.append(1)
            for val in mean[0]:
                if val > 0.5:
                    fea_list.append(1)
                else:
                    fea_list.append(0)
            if type == 1 or type == 3:
                fea_list.append(1)
            feature_Select = self.numtofea(fea_list, self.Feature)  # 将0，1串映射到特征的选取中
            data_sample = self.read_data_fea(feature_Select, self.data)
            sample_name = 'E:/Vehicle%d.txt' % i
            self.saveData1(sample_name, np.array(data_sample))

    def _readData(self, filename, dataType, workersize):
        '''
        :param filename: 数据文件名
        :param dataType: 数据的保存方式,1表示空格最后；2表示空格最前；3表示逗号最后；4表示逗号最前
        :return:
        '''
        # 数据由空格划分，标签在最后一列
        if dataType == 1:
            numFeat = len(open(filename).readline().split(' '))
            self.size = numFeat
            self.DNA_size = self.size - 1
            fr = open(filename)
            for line in fr.readlines():
                xi = []
                curline = line.strip().split(' ')
                for i in range(numFeat):
                    xi.append((curline[i]))
                self.data.append(xi)
            self.dividedata(workersize, 1)
        # 数据由空格划分，标签在第一列
        if dataType == 2:
            numFeat = len(open(filename).readline().split(' '))
            self.size = numFeat
            self.DNA_size = self.size - 1
            fr = open(filename)
            for line in fr.readlines():
                xi = []
                curline = line.strip().split(' ')
                for i in range(numFeat):
                    xi.append((curline[i]))
                self.data.append(xi)
            self.dividedata(workersize, 2)
        if dataType == 3:
            numFeat = len(open(filename).readline().split(','))
            self.size = numFeat
            self.DNA_size = self.size - 1
            fr = open(filename)
            for line in fr.readlines():
                xi = []
                curline = line.strip().split(',')
                for i in range(numFeat):
                    xi.append((curline[i]))
                self.data.append(xi)
            self.dividedata(workersize, 3)
        if dataType == 4:
            numFeat = len(open(filename).readline().split(','))
            self.size = numFeat
            self.DNA_size = self.size - 1
            fr = open(filename)
            for line in fr.readlines():
                xi = []
                curline = line.strip().split(',')
                for i in range(numFeat):
                    xi.append((curline[i]))
                self.data.append(xi)
            self.dividedata(workersize, 4)

class LoadData(object):
    def __init__(self):
        self.DNA_size = 0
        self.Feature = []
        self.trainX = []
        self.trainy = []
        self.testX = []
        self.testy = []

    def _readData(self, readType, filename, dataType, test_size):
        '''
        :param readType: 数据集的划分方式
        :param filename: 数据文件名
        :param dataType: 数据的保存方式,1表示空格最后；2表示空格最前；3表示逗号最后；4表示逗号最前
        :return:
        '''
        # 如果是通过随机划分的数据
        self.readType = readType
        self.test_size = test_size
        if readType == 'split':
            # 数据由空格划分，标签在最后一列
            if dataType == 1:
                data = pd.read_table(filename, sep=' ')
                x, y = data.ix[:, 0:len(open(filename).readline().split(' ')) - 1], data.ix[:, -1]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                self.getMatDataBysplit(x_train, x_test, y_train, y_test)
            # 数据由空格分离，标签在第一列
            elif dataType == 2:
                data = pd.read_table(filename, sep=' ')
                x, y = data.ix[:, 1:], data.ix[:, 0]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                self.getMatDataBysplit(x_train, x_test, y_train, y_test)
            # 数据由逗号分离，标签在最后一列
            elif dataType == 3:
                data = pd.read_table(filename, sep=',')
                x, y = data.ix[:, 0:len(open(filename).readline().split(',')) - 1], data.ix[:, -1]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                self.getMatDataBysplit(x_train, x_test, y_train, y_test)
            # 数据由逗号分离，标签在第一列
            elif dataType == 4:
                data = pd.read_table(filename, sep=',')
                x, y = data.ix[:, 1:], data.ix[:, 0]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                self.getMatDataBysplit(x_train, x_test, y_train, y_test)
        # 如果是通过N-折的方式进行划分
        elif readType == 'nFold':
            # 特征空格，标签在最后一列
            if dataType == 1:
                numFeat = len(open(filename).readline().split(' ')) - 1
                feature = []
                label = []
                fr = open(filename)
                for line in fr.readlines():
                    xi = []
                    curline = line.strip().split(' ')
                    for i in range(numFeat):
                        xi.append(float(curline[i]))
                    feature.append(xi)
                    label.append((curline[-1]))
                skf = KFold(n_splits=test_size)
                skf.get_n_splits(feature, label)
                self.getMatDataBynfold(skf, feature, label)
            # 特征空格，标签第一列
            elif dataType == 2:
                numFeat = len(open(filename).readline().split(' '))
                feature = []
                label = []
                fr = open(filename)
                for line in fr.readlines():
                    xi = []
                    curline = line.strip().split(' ')
                    label.append(float(curline[0]))
                    for i in range(1, numFeat):
                        xi.append(float(curline[i]))
                    feature.append(xi)
                skf = KFold(n_splits=test_size)
                skf.get_n_splits(feature, label)
                self.getMatDataBynfold(skf, feature, label)
            # 特征逗号，标签在最后一列
            elif dataType == 3:
                numFeat = len(open(filename).readline().split(',')) - 1
                feature = []
                label = []
                fr = open(filename)
                for line in fr.readlines():
                    xi = []
                    curline = line.strip().split(',')
                    for i in range(numFeat):
                        xi.append(float(curline[i]))
                    feature.append(xi)
                    label.append((curline[-1]))
                skf = KFold(n_splits=test_size)
                skf.get_n_splits(feature, label)
                self.getMatDataBynfold(skf, feature, label)
            # 特征由逗号，标签在第一列
            elif dataType == 4:
                numFeat = len(open(filename).readline().split(','))
                feature = []
                label = []
                fr = open(filename)
                for line in fr.readlines():
                    xi = []
                    curline = line.strip().split(',')
                    label.append(float(curline[0]))
                    for i in range(1, numFeat):
                        xi.append(float(curline[i]))
                    feature.append(xi)
                skf = KFold(n_splits=test_size)
                skf.get_n_splits(feature, label)
                self.getMatDataBynfold(skf, feature, label)

    # 将列表形式的数据转换为矩阵形式
    def getMatDataBysplit(self, x_train, x_test, y_train, y_test):
        self.trainX.append(np.array(x_train))
        self.testX.append(np.array(x_test))
        self.trainy.append(np.array(y_train))
        self.testy.append(np.array(y_test))
        self.DNA_size = len(np.array(x_train)[0])
        # self.saveData1('E:/trainX.txt', np.array(x_train))
        # self.saveData1('E:/predictX.txt', np.array(x_test))
        # self.saveData2('E:/trainy.txt', np.array(y_train))
        # self.saveData2('E:/predicty.txt', np.array(y_test))
        # print('zhangxinzhangxin', self.trainX[0], self.testX[0], self.trainy, self.testy)
        for i in range(self.DNA_size):
            self.Feature.append(i)  # 其中的元素是特征所在的位置

    def getMatDataBynfold(self, skf, feature, label):
        for train_index, test_index in skf.split(feature, label):
            self.trainX.append(np.array(feature)[train_index])
            self.testX.append(np.array(feature)[test_index])
            self.trainy.append(np.array(label)[train_index])
            self.testy.append(np.array(label)[test_index])
        self.DNA_size = len(np.array(feature)[0])
        for i in range(self.DNA_size):
            self.Feature.append(i)

    def saveData1(self, filename, dataname):
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

    def saveData2(self, filename, dataname):
        with open(filename, 'w') as file_object:  # 将文件及其内容存储到变量file_object

            # 写入第一行(第一块)
            file_object.write(str(dataname[0]))  # 写第一行第一列

            # 写入第一行后面的行（第二块）
            for i in range(1, np.size(dataname, 0)):
                file_object.write('\n' + str(dataname[i]))

    def getDNA_size(self):
        return self.DNA_size

    def getdata(self):
        return self.trainX, self.testX, self.trainy, self.testy

    def getFeature(self):
        return self.Feature

    def getReadType(self):
        return self.readType

    def getTestsize(self):
        return self.test_size

'''
然后就是对每个线程工作内容的定义
'''

class Worker(object):
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.LR = LEARNING_RATE
        self.popnet = Worker_pop(name, data)  # 每个线程在初始化中首先区初始化一个网络

    def numtofea(self, num, fea_list):
        feature = []
        for i in range(len(num)):
            if num[i] == 1:
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
            clf = neighbors.KNeighborsClassifier(n_neighbors=5)  # 创建分类器对象
            if (len(data_train[0]) > 0):
                clf.fit(data_train, label_train)  # 用训练数据拟合分类器模型搜索
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, label_pre)
        elif train_cla is 'svm':
            clf = svm.SVC()
            if (len(data_train[0]) > 0):
                clf.fit(data_train, label_train)
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, label_pre)
        elif train_cla is 'tree':
            if (len(data_train[0]) > 0):
                clf = DecisionTreeClassifier()
                clf.fit(data_train, label_train)
                predict = clf.predict(data_pre)
                acc = self.acc_pre(predict, label_pre)
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
        return (1 - num / len(label_train))

    def dr_pre(self, feature_list):
        feature_sum = len(feature_list)
        num = didata.getDNA_size()
        count = 0
        for i in feature_list:
            if (i == 1):
                count = count + 1
        return 1 - (count / num)

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

    def work(self):
        for g in range(MAX_GLOBAL_EP):
            if g % 10 == 0:
                lock_max_fit.acquire()
                max_fit = global_pop.getMaxfit()
                lock_max_fit.release()
                print(self.name, g+1, '当前最大值：', max_fit)
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
            # kids = self.popnet._make_kids()
            kids_fit = []  # 初始化子代的适应度值，这是一个列表，初始化为空
            feature_set = []  # 所有子代选取特征的集合
            kids_fits = []
            for kid in kids:  # 遍历每一个子代
                feature_list = []
                for feature in kid:  # 遍历每个子代的每个特征
                    if feature > 0.0:  # 对于特征值大于0.5的特征就用1表示
                        feature_list.append(1)
                    else:
                        feature_list.append(0)  # 否则就用0来表示
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
                    kid_fit = self.get_fitness(data_sample, trainy, data_predict, testy, 'train_knn')  # 然后更具所选的特征集合进行准确率求解
                    lock_max_fit.acquire()
                    max = global_pop.getMaxfit()
                    lock_max_fit.release()
                    kid_tifs = kid_fit - max
                    kids_fit.append(kid_tifs)
                    kids_fits.append(kid_fit)  # 然后将每一的子代的适应度添加到适应度的集合当中
                elif self.data.getReadType() is 'nFold':
                    kid_fit = 0.0
                    for i in range(self.data.getTestsize()):
                        trainX_, testX_, trainy_, testy_ = self.data.getdata()
                        trainX = trainX_[i]
                        testX = testX_[i]
                        trainy = trainy_[i]
                        testy = testy_[i]
                        data_sample = self.read_data_fea(feature_Select, trainX)
                        data_predict = self.read_data_fea(feature_Select, testX)  # 找到所选特征所对应的数据集
                        kid_fit_sample = self.get_fitness(data_sample, trainy, data_predict, testy, 'train_knn')
                        kid_fit = kid_fit + kid_fit_sample
                    kid_fit = kid_fit / self.data.getTestsize()
                    lock_max_fit.acquire()
                    max = global_pop.getMaxfit()
                    lock_max_fit.release()
                    kid_tifs = kid_fit - max
                    kids_fit.append(kid_tifs)
                    kids_fits.append(kid_fit)  # 然后将每一的子代的适应度添加到适应度的集合当中
            sess.run(self.popnet.train_op, {self.popnet.tfkids_fit: kids_fit, self.popnet.tfkids: kids}) # 然后根据所有子代的适应度来进行参数更新
            new_max, count = self.get_max(kids_fits)  # 找到子代中准确率最高的孩子，以及返回孩子的位置
            feature_get = feature_set[count]  # 根据找到的位置找到相应的特征选取
            new_dr = self.dr_pre(feature_get)  # 然后更具特征选取来求出所对应的维度缩减率
            changed = self.insert_value(new_max, new_dr, 1)  # 将新得到的值一目前最大值进行比较
            if changed is True:
                print('   ', self.name, '使用', 'knn', '第', g + 1, '轮迭代：')
                print('选取特征为：', feature_get)
                print('维度缩减为：', new_dr)
                print('准确率为：', new_max)
                print(kids_fits)
                # self.popnet._update_net()
                # print(self.name, g+1, '更新全局网络为：', sess.run(self.popnet.mvn.mean()))
            # if g % 20 == 0 and g!=0:
            #     self.popnet._pull_net()
                # print(self.name, g+1, '获取目标网络为：', sess.run(self.popnet.mvn.mean()))

if __name__ == '__main__':
    # 首先这是种群中子代的数目，也就是说每一次迭代的数量
    N_POP = 50
    # 然后定义网络更新的学习率
    LEARNING_RATE = 0.001
    # 然后就是定义算法的迭代次数
    MAX_GLOBAL_EP = 1000
    # 之后就是初始化一个回话来运行网络
    sess = tf.Session()
    # 然后是首先创建全局网络
    # global_pop = Global_pop('global')
    # 创建读取数据的工具类
    # 首先进行数据集的划分
    N_WORKERS = multiprocessing.cpu_count()
    didata = divideData()
    didata._readData('E:/arcene15.txt', 1, N_WORKERS)
    data = []
    for i in range(N_WORKERS):
        data.append(LoadData())
        filename = 'E:/Vehicle%d.txt' % i
        data[i]._readData('split', filename, 1, 0.3)
    print('zhangxin')
    print(data)
    print(data[2].getFeature())
    # # 首先进行数据集的划分
    # N_WORKERS = multiprocessing.cpu_count()
    # data.divide_data(N_WORKERS)
    global_pop = Global_pop('global', didata)
    #接着就是定义一些所来对共性资源进行锁定
    lock_kids = threading.Lock()
    lock_max_fit = threading.Lock()
    lock_dr = threading.Lock()
    lock_push = threading.Lock()
    lock_pull = threading.Lock()
    #接着就是通过核心数来创建多线程进行工作
    with tf.device("/cpu:0"):
        workers = []
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            idata = data[i]
            print(idata.getFeature())
            workers.append(Worker(i_name, idata))

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
