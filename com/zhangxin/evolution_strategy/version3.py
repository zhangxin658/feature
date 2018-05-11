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

GLOBAL_NET_SCOPE = 'Global_Net'  #全局网络

class POPNet(object):
  def __init__(self, scope, globlPOP=None):
    if scope == GLOBAL_NET_SCOPE:
      self.max_fit = 0
      self.dr = 0
      with tf.variable_scope(scope):
        self.mvn = self._built_net(scope)
        self.tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
        self.tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
        self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * self.tfkids_fit * 1.2 + 0.01 * self.mvn.log_prob(self.tfkids) * self.mvn.prob(self.tfkids))
    else:
      with tf.variable_scope(scope):
        self.LR = 0
        self.max_fit = 0
        self.mvn = self._built_net(scope)
        self.tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
        self.tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
        self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * (self.tfkids_fit-self.max_fit) * 1.2 + 0.01 * self.mvn.log_prob(self.tfkids) * self.mvn.prob(self.tfkids))
        self.train_op = tf.train.GradientDescentOptimizer(self.LR).minimize(self.loss)

  def _built_net(self, scope):
    mean = tf.Variable(tf.truncated_normal([DNA_SIZE, ], stddev=0.02, mean=0.5), dtype=tf.float32, name=scope+'_mean')
    cov = tf.Variable(0.1 * tf.eye(DNA_SIZE), dtype=tf.float32, name=scope+'_cov')
    mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=abs(
      cov + tf.Variable(0.0001 * tf.eye(DNA_SIZE), dtype=tf.float32)), name=scope)
    sess.run(tf.global_variables_initializer())
    return mvn

  def _run_net(self, LR, kids_fit, kids, max_fit):
    # self.train_op = tf.train.GradientDescentOptimizer(LR).minimize(self.loss)  # compute and apply gradients for mean and cov
    self.LR = LR
    self.max_fit = max_fit
    sess.run(self.train_op, {self.tfkids_fit: kids_fit, self.tfkids: kids})

  def _update_net(self):
    lock_update.acquire()
    pop.mvn = self.mvn.copy()
    lock_update.release()

  def _pull_net(self):
    lock_pull.acquire()
    self.mvn = pop.mvn.copy()
    lock_pull.release()


  def _get_net(self):
    return self.mvn

  def _get_kids(self, lock):
    self._make_kids(lock)
    return self.kids

  def _make_kids(self, lock):
    lock.acquire()
    self.kids = sess.run(self.mvn.sample(N_POP))
    lock.release()

  def _get_fit(self):
    return self.max_fit

  def _set_fit(self, max_fit):
    self.max_fit = max_fit

  def _get_dr(self):
    return self.dr

  def _set_dr(self, dr):
    self.dr = dr


class Worker(object):
  def __init__(self, name):
    self.name = name
    self.popnet = POPNet(name, pop)
    self.LR = LEARNING_RATE
    self.feature_set = []  #所有孩子特征的集合

  def work(self, lock):
    for g in range(MAX_GLOBAL_EP):
      if g % 10 == 0:
        self.LR = self.LR * 0.9
      kids = self.popnet._get_kids(lock)
      self.kids_fit = []
      kids_fit_minus = []
      #print(self.name, kids)
      for kid in kids:
        feature_list = []
        for feature in kid:
          if feature > 0.5:
            feature_list.append(1)
          else:
            feature_list.append(0)
        feature_Select = self.numtofea(feature_list, Feature)
        self.feature_set.append(feature_list)
        # print('===========================输出选取的特征==============================')
        # print(feature_Select)
        data_sample = self.read_data_fea(feature_Select, trainX)
        data_predict = self.read_data_fea(feature_Select, predictX)
        # print('===========================输出训练集和预测集============================')
        # print(data_sample)
        # print(data_predict)
        kid_fit = self.get_fitness(data_sample, trainy, data_predict, predicty, SKL)
        self.kids_fit.append(kid_fit)
      lock_max_fit.acquire()
      max_fit = pop._get_fit()
      lock_max_fit.release()
      self.popnet._run_net(self.LR, self.kids_fit, kids, max_fit)
      new_max, count = self.get_max(self.kids_fit)
      feature_get = self.feature_set[count]
      lock_max_fit.acquire()
      max_fit = pop._get_fit()
      lock_max_fit.release()
      if (new_max > max_fit):
        lock_max_fit.acquire()
        pop._set_fit(new_max)
        lock_max_fit.release()
        dr = self.dr_pre(feature_get)
        lock_dr.acquire()
        pop._set_dr(dr)
        lock_dr.release()
        print('   ', self.name, '使用', SKL, '第', g + 1, '轮迭代：')
        print('选取特征为：', feature_get)
        print('维度缩减为：', dr)
        print('准确率为：', new_max)
        print(self.kids_fit)
        self.popnet._update_net()
      elif (new_max == max_fit):
        new_dr = self.dr_pre(feature_get)
        lock_dr.acquire()
        dr = pop._get_dr()
        lock_dr.release()
        if (new_dr > dr):
          lock_dr.acquire()
          pop._set_dr(new_dr)
          lock_dr.release()
          print('   ', self.name, '使用', SKL, '第', g + 1, '轮迭代：')
          print('选取特征为：', feature_get)
          print('维度缩减为：', new_dr)
          print('准确率为：', new_max)
          print(self.kids_fit)
          self.popnet._update_net()
    if (g % 10 ==0):
      self.popnet.pull()



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
    return acc * 100

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

if __name__ == '__main__':
  N_POP = 2  #种群大小，可改变，根据数据集大小动态设定
  LEARNING_RATE = 0.02   #学习率，可以改变，初始为0.02
  MAX_GLOBAL_EP = 20    #迭代次数，可改变
  sess = tf.Session()
  train_X, predict_X, train_y, predict_y = loadData_split('E:/Sonar.txt', 3, 1, 1)
  trainX = np.array(train_X)
  predictX = np.array(predict_X)
  trainy = np.array(train_y)
  predicty = np.array(predict_y)
  lock = threading.Lock()
  lock_max_fit = threading.Lock()
  lock_update = threading.Lock()
  lock_pull = threading.Lock()
  lock_dr = threading.Lock()
  DNA_SIZE = len(trainX[0])
  print('============================输出数据集============================')
  print(trainX, predictX, trainy, predicty)
  pop = POPNet(GLOBAL_NET_SCOPE)
  # print('============================输出子代==============================')
  # print(pop._get_kids(lock))
  num_fea_original = mat(trainX).shape[1]
  Feature = []
  for i in range(num_fea_original):
    Feature.append(i)
  # print('============================输出特征==============================')
  # print(Feature)
  N_WORKERS = multiprocessing.cpu_count()
  # print('============================输出种群数量===========================')
  # print(N_WORKERS)
  with tf.device("/cpu:0"):
    GLOBAL_POP = POPNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    for i in range(N_WORKERS):
      i_name = 'W_%i' % i  # worker name
      workers.append(Worker(i_name))

  COORD = tf.train.Coordinator()
  sess.run(tf.global_variables_initializer())

  worker_threads = []
  for worker in workers:
    job = lambda: worker.work(lock)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
  COORD.join(worker_threads)