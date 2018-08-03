from multiprocessing import Process
import multiprocessing
from numpy import *#mat
import threading
from sklearn import neighbors
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

# =============================
# 这是一些初始化的参数
N_WORKERS = multiprocessing.cpu_count()  #返回计算机核心数，也就是并行进程的个数
MAX_GLOBAL_EP = 1000  #最大的迭代次数
GLOBAL_NET_SCOPE = 'Global_Net'  #全局网络
UPDATE_GLOBAL_ITER = 10  #更新时间步
ENTROPY_BETA = 0.01
LEARNING_RATE = 0.02  #学习率
N_POP = 50   #种群个数

# ============================
# 这个 class 可以被调用生成一个 global net.
# 也能被调用生成一个 worker 的 net, 因为他们的结构是一样的,
# 所以这个 class 可以被重复利用.

class POPNet(object):
    def __init__(self, scope, globalPOP=None):
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.mvn = self._built_net(scope, DNA_SIZE, N_POP)
        else:
            with tf.variable_scope(scope):
                self.tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
                self.tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
                self.mvn = self._built_net(scope, DNA_SIZE, N_POP)

                with tf.name_scope('make_kid'):
                    self.mvn.sample(N_POP)

                with tf.name_scope('loss'):
                    log_prob = self.mvn.log_prob(self.tfkids)
                    exp_v = log_prob * self.tfkids_fit
                    entropy = 0.01 * self.mvn.log_prob(self.tfkids + 1e-5) * self.mvn.prob(self.tfkids)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = -tf.reduce_mean(self.exp_v)

                with tf.name_scope('local_grad'):
                    self.train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
                        self.a_loss)  # compute and apply gradients for mean and cov

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.mvn = globalPOP.mvn
                    print(self.mvn.mean)
                    # self.mvn = tf.Variable(globalPOP)
                    # self.pull_stdd_op = [l_p.assign(g_p) for l_p, g_p in zip(self.mvn.stddev(), globalPOP.mvn.stddev())]
                with tf.name_scope('push'):
                    # self.update_pop = self.train_op.apply_gradients((self.train_op, globalPOP.mvn))
                    globalPOP.mvn = self.mvn

    def _built_net(self, scope, DNA_SIZE, N_POP):
        mean = tf.Variable(tf.truncated_normal([DNA_SIZE, ], stddev=0.02, mean=0.5), dtype=tf.float32)
        cov = tf.Variable(tf.eye(DNA_SIZE), dtype=tf.float32)
        mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=abs(
             cov + tf.Variable(0.001 * tf.eye(DNA_SIZE), dtype=tf.float32)))
        # make_kid = mvn.sample(N_POP)
        return mvn

    def update_global(self, globalPOP):  # run by a local
        # SESS.run([self.update_pop], feed_dict)  # local grads applies to global net
        globalPOP.mvn = self.mvn
        print('执行更新')

    def pull_global(self, globalPOP):  # run by a local
        SESS.run([self.pull_params_op])
        print('执行下载')


    def get_kids(self):
        return SESS.run(self.mvn.sample(N_POP))

    def pop_run(self, kids_fit, kids):
        SESS.run(self.train_op, {self.tfkids_fit: kids_fit, self.tfkids: kids})




class Worker(object):

    def __init__(self, name, globalPOP, feature):
        self.name = name
        self.POPNet = POPNet(name, globalPOP)
        self.trainX = trainX
        self.predictX = predictX
        self.trainy = trainy
        self.predicty = predicty
        self.LR = LEARNING_RATE
        self.feature = feature

    def work(self, globalPOP):
        max = 0
        for g in range(MAX_GLOBAL_EP):
            if MAX_GLOBAL_EP % 10 == 0:
                self.LR = self.LR * 0.9
            kids = self.POPNet.get_kids()
            self.kids_fit = []
            feature_set = []
            for i in kids:
                feature_list = []
                k = 0
                for j in i:
                    if j > 0.5:
                        feature_list.append(1)
                    else:
                        feature_list.append(0)
                fea_list_CB = self.numtofea(feature_list, self.feature)
                feature_set.append(feature_list)
                data_sample = self.read_data_fea(fea_list_CB, trainX)
                data_predict = self.read_data_fea(fea_list_CB, predictX)
                kid_fit = self.get_fitness(data_sample, trainy, data_predict, predicty, SKL) * 100 - max
                # print(kid_fit)
                self.kids_fit.append(kid_fit)
            self.POPNet.pop_run(self.kids_fit, kids)
            # sess.run(self.POPNet.train_op, {self.POPNet.tfkids_fit: kids_fit, self.POPNet.tfkids: kids})  # update distribution parameters
            new_max, count = self.get_max(self.kids_fit)
            new_max = new_max + max
            feature_get = feature_set[count]
            if (new_max > max):
                max = new_max
                dr = self.dr_pre(feature_get)
                print('   ', self.name, '使用', SKL, '第', g + 1, '轮迭代：')
                print('选取特征为：', feature_get)
                print('维度缩减为：', dr)
                print('准确率为：', new_max)
                print(self.kids_fit)
                self.POPNet.update_global(globalPOP)
            elif (new_max == max):
                new_dr = self.dr_pre(feature_get)
                if (new_dr > dr):
                    dr = new_dr
                    print('   ', self.name, '使用', SKL, '第', g + 1, '轮迭代：')
                    print('选取特征为：', feature_get)
                    print('维度缩减为：', dr)
                    print('准确率为：', max_fit.getmax_fit())
                    print(self.kids_fit)
                    self.POPNet.update_global(globalPOP)
        self.POPNet.pull_global(globalPOP)

    def numtofea(self, num, fea_list):
        feature = []
        for i in range(len(num)):
            if num[i] == 1:
                feature.append(fea_list[i])
            else:
                continue
        return feature

    def read_data_fea(self, fea_list, dataset):
        dataMat = mat(dataset)
        col = dataMat.shape[0]  # 行号
        data_sample = []
        for i in range(col):
            col_i = []
            for j in fea_list:
                col_i.append(dataMat[i, j])
            data_sample.append(col_i)
        return data_sample

    def get_fitness(self, data_train, label_train, data_pre, label_pre, train_cla):
        acc = 0
        if train_cla is 'train_knn':
            clf = neighbors.KNeighborsClassifier(n_neighbors=K_NN_type)  # 创建分类器对象
            if (len(data_train[0]) > 0):
                clf.fit(data_train, label_train)  # 用训练数据拟合分类器模型搜索
                predict = clf.predict(data_pre)
                num = 0
                for i in range(len(label_pre)):
                    if predict[i] != label_pre[i]:
                        num += 1
                acc = (1 - num / len(label_train))
                # acc = self.acc_pre(predict, label_pre)  # 预测标签和ground_true标签对比 算准确率
        elif train_cla is 'svm':
            clf = svm.SVC()
            if (len(data_train[0]) > 0):
                clf.fit(data_train, label_train)
                predict = clf.predict(data_pre)
                num = 0
                for i in range(len(label_pre)):
                    if predict[i] != label_pre[i]:
                        num += 1
                acc = (1 - num / len(label_train))
        return acc

    def get_max(self, new_list):
        max = -101
        count = 0
        for i in new_list:
            count = count + 1
            if (i > max):
                max = i
        return max, count - 1

    def acc_pre(self, label_pre, label_train):
        num = 0
        for i in range(len(label_pre)):
            if label_pre[i] != label_train[i]:
                num += 1
        return (1 - num / len(label_train))

    def dr_pre(self, feature_list):
        feature_sum = len(feature_list)

        count = 0
        for i in feature_list:
            if (i == 1):
                count = count + 1
        return 1 - (count / feature_sum)



def loadData_split(filename, type, k_nn, skl):
    global K_NN_type
    global SKL
    K_NN_type = k_nn
    if (skl == 1):
        SKL = 'train_knn'
    elif (skl == 2):
        SKL = 'svm'
    # 数据由空格分离，标签在最后一列
    if (type == 1):
        data = pd.read_table(filename, sep=' ')
        x, y = data.ix[:, 0:len(open(filename).readline().split(' ')) - 1], data.ix[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    # 数据由空格分离，标签在第一列
    elif (type == 2):
        data = pd.read_table(filename, sep=' ')
        x, y = data.ix[:, 1:], data.ix[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    # 数据由逗号分离，标签在最后一列
    elif (type == 3):
        data = pd.read_table(filename, sep=',')
        x, y = data.ix[:, 0:len(open(filename).readline().split(',')) - 1], data.ix[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    # 数据由逗号分离，标签在第一列
    elif (type == 4):
        data = pd.read_table(filename, sep=',')
        x, y = data.ix[:, 1:], data.ix[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test

class Max_fit(object):
    def __init__(self, max):
        self.max_fit = 0
    def getmax_fit(self):
        return self.max_fit
    def setmax_fit(self, new_fit):
        self.max_fit = new_fit

if __name__ == "__main__":
    SESS = tf.Session()
    train_X, predict_X, train_y, predict_y = loadData_split('E:/Sonar.txt', 3, 1, 1)
    trainX = np.array(train_X)
    predictX = np.array(predict_X)
    trainy = np.array(train_y)
    predicty = np.array(predict_y)
    max_fit = Max_fit(0)
    print(trainX, predictX, trainy, predicty)
    DNA_SIZE = len(trainX[0])
    num_fea_original = mat(trainX).shape[1]
    feature = []
    for i in range(num_fea_original):
        feature.append(i)
    print(feature)
    LEARNING_RATE = 0.02

    with tf.device("/cpu:0"):
        # OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_POP = POPNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_POP, feature))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(GLOBAL_POP)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)