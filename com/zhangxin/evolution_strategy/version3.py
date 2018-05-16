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
        self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * (self.tfkids_fit-self.max_fit) + 0.01 * self.mvn.log_prob(self.tfkids) * self.mvn.prob(self.tfkids))
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
    cov = tf.Variable(1.0 * tf.eye(DNA_SIZE), dtype=tf.float32, name=scope+'_cov')
    mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=abs(cov) + tf.Variable(0.0001 * tf.eye(DNA_SIZE), dtype=tf.float32), name=scope)
    sess.run(tf.global_variables_initializer())
    return mvn

  def _run_net(self, LR, kids_fit, kids):
    # self.train_op = tf.train.GradientDescentOptimizer(LR).minimize(self.loss)  # compute and apply gradients for mean and cov
    self.LR = LR
    lock_max_fit.acquire()
    max_fit = pop._get_fit()
    lock_max_fit.release()
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

  def _get_kids(self):
    self._make_kids()
    return self.kids

  def _make_kids(self):
    lock_thread.acquire()
    self.kids = sess.run(self.mvn.sample(N_POP))
    lock_thread.release()

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
    # self.feature_set = []  #所有孩子特征的集合

  def work(self):
    for g in range(MAX_GLOBAL_EP):
      if g % 10 == 0:
        self.LR = self.LR * pow(0.9, g/10)  #每迭代五次就使学习率减小
      kids = self.popnet._get_kids()  #生成子代
      self.kids_fit = []     #初始化子代的适应度值，这是一个列表，初始化为空
      self.feature_set = []    #所有子代选取特征的集合
      #print(self.name, kids)
      for kid in kids:         #遍历每一个子代
        feature_list = []
        for feature in kid:     #遍历每个子代的每个特征
          if feature > 0.50:        #对于特征值大于0.5的特征就用1表示
            feature_list.append(1)
          else:
            feature_list.append(0)    #否则就用0来表示
        feature_Select = self.numtofea(feature_list, Feature)   #将0，1串映射到特征的选取中
        self.feature_set.append(feature_list)       #将每个子代所选取的特征放到特征集合中
        # print('===========================输出选取的特征==============================')
        # print(feature_Select)
        data_sample = self.read_data_fea(feature_Select, trainX)
        data_predict = self.read_data_fea(feature_Select, predictX)    #找到所选特征所对应的数据集
        # print('===========================输出训练集和预测集============================')
        # print(data_sample)
        # print(data_predict)
        kid_fit = self.get_fitness(data_sample, trainy, data_predict, predicty, SKL)   #然后更具所选的特征集合进行准确率求解
        self.kids_fit.append(kid_fit)     #然后将每一的子代的适应度添加到适应度的集合当中
      # lock_max_fit.acquire()
      # max_fit = pop._get_fit()
      # lock_max_fit.release()
      self.popnet._run_net(self.LR, self.kids_fit, kids)    #然后根据所有子代的适应度来进行参数更新
      new_max, count = self.get_max(self.kids_fit)  #找到子代中准确率最高的孩子，以及返回孩子的位置
      feature_get = self.feature_set[count]      #根据找到的位置找到相应的特征选取
      new_dr = self.dr_pre(feature_get)        #然后更具特征选取来求出所对应的维度缩减率
      changed = self.insert_value(new_max, new_dr, 1)      #将新得到的值一目前最大值进行比较
      if changed is True:
        print('   ', self.name, '使用', SKL, K_NN_type, '第', g + 1, '轮迭代：')
        print('选取特征为：', feature_get)
        print('维度缩减为：', new_dr)
        print('准确率为：', new_max)
        print(self.kids_fit)
        self.popnet._update_net()
    if (g % 15 ==0):
      self.popnet._pull_net()

  def insert_value(self, new_max, new_dr, bili):
    lock_max_fit.acquire()
    max_fit = pop._get_fit()
    lock_max_fit.release()
    lock_dr.acquire()
    dr = pop._get_dr()
    lock_dr.release()
    if (max_fit * bili + dr * 100 * (1 - bili)) < (new_max * bili + new_dr * 100 * (1 - bili)):
      lock_max_fit.acquire()
      pop._set_fit(new_max)
      lock_max_fit.release()
      lock_dr.acquire()
      pop._set_dr(new_dr)
      lock_dr.release()
      return True
    else:
      return False

  def numtofea(self, num, fea_list):
    feature = []
    for i in range(len(num)):
      if num[i] == 1:
        feature.append(fea_list[i])
      else:
        continue
    return feature   #返回的是所选特征所在的位置

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
    acc = 0.00
    if train_cla is 'train_knn':
      clf = neighbors.KNeighborsClassifier(n_neighbors=K_NN_type)  # 创建分类器对象
      if (len(data_train[0]) > 0):
        clf.fit(data_train, label_train)  # 用训练数据拟合分类器模型搜索
        predict = clf.predict(data_pre)
        acc = self.acc_pre(predict, label_pre)
        # num = 0
        # for i in range(len(label_pre)):
        #   if predict[i] != label_pre[i]:
        #     num += 1
        # acc = (1 - num / len(label_train))
        # acc = self.acc_pre(predict, label_pre)  # 预测标签和ground_true标签对比 算准确率
    elif train_cla is 'svm':
      clf = svm.SVC()
      if (len(data_train[0]) > 0):
        clf.fit(data_train, label_train)
        predict = clf.predict(data_pre)
        acc = self.acc_pre(predict, label_pre)
        # num = 0
        # for i in range(len(label_pre)):
        #   if predict[i] != label_pre[i]:
        #     num += 1
        # acc = (1 - num / len(label_train))
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
    return max, index - 1   #返回最大的适应的以及所对应的子代

  def acc_pre(self, predict, label_train):
    num = 0
    for i in range(len(predict)):
      if predict[i] != label_train[i]:
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

def saveData1(filename, dataname):
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

def saveData2(filename, dataname):
  with open(filename, 'w') as file_object:  # 将文件及其内容存储到变量file_object

    # 写入第一行(第一块)
    file_object.write(str(dataname[0]))  # 写第一行第一列

    # 写入第一行后面的行（第二块）
    for i in range(1, np.size(dataname, 0)):
      file_object.write('\n' + str(dataname[i]))


if __name__ == '__main__':
  N_POP = 50  #种群大小，可改变，根据数据集大小动态设定
  LEARNING_RATE = 0.5   #学习率，可以改变，初始为0.02
  MAX_GLOBAL_EP = 1000    #迭代次数，可改变
  sess = tf.Session()
  train_X, predict_X, train_y, predict_y = loadData_split('E:/wine.txt', 4, 1, 1)
  trainX = np.array(train_X)
  predictX = np.array(predict_X)
  trainy = np.array(train_y)
  predicty = np.array(predict_y)
  lock_thread = threading.Lock()
  lock_max_fit = threading.Lock()
  lock_update = threading.Lock()
  lock_pull = threading.Lock()
  lock_dr = threading.Lock()
  DNA_SIZE = len(trainX[0])
  print('============================输出数据集============================')
  saveData1('E:/trainX.txt', trainX)
  saveData1('E:/predictX.txt', predictX)
  saveData2('E:/trainy.txt', trainy)
  saveData2('E:/predicty.txt', predicty)
  print(trainX, predictX, trainy, predicty)
  pop = POPNet(GLOBAL_NET_SCOPE)
  # print('============================输出子代==============================')
  # print(pop._get_kids(lock))
  num_fea_original = mat(trainX).shape[1]
  Feature = []
  for i in range(num_fea_original):
    Feature.append(i)  # 其中的元素是特征所在的位置
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
    job = lambda: worker.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
  COORD.join(worker_threads)