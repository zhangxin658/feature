from numpy import *#mat
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

#=============================================加载文件并初始化特征========================================

def loadData(filename):
    numFeat=len(open(filename).readline().split(','))-1
    # print(numFeat)
    feature=[]
    label=[]
    fr=open(filename)
    for line in fr.readlines():
        xi=[]
        curline=line.strip().split(',')
        label.append(float(curline[0]))
        for i in range(1, numFeat + 1):
            xi.append(float(curline[i]))
        feature.append(xi)
        #label.append(float(curline[-1]))
    return feature,label

def loadData_tail(filename):
    numFeat=len(open(filename).readline().split(','))-1
    feature=[];label=[]
    fr=open(filename)
    for line in fr.readlines():
        xi=[]
        curline=line.strip().split(',')
        for i in range(numFeat):
            xi.append(float(curline[i]))
        feature.append(xi)
        label.append((curline[-1]))
    return feature,label

def loadData_split(filename, type, k_nn, skl):
    global K_NN_type
    global SKL
    K_NN_type = k_nn
    if(skl == 1):
        SKL = 'train_knn'
    elif(skl == 2):
        SKL = 'svm'
    #数据由空格分离，标签在最后一列
    if(type == 1):
        data = pd.read_table(filename, sep=' ')
        x, y = data.ix[:, 0:len(open(filename).readline().split(' ')) - 1], data.ix[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    #数据由空格分离，标签在第一列
    elif(type == 2):
        data = pd.read_table(filename, sep=' ')
        x, y = data.ix[:, 1:], data.ix[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    #数据由逗号分离，标签在最后一列
    elif(type == 3):
        data = pd.read_table(filename, sep=',')
        x, y = data.ix[:, 0:len(open(filename).readline().split(',')) - 1], data.ix[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
    #数据由逗号分离，标签在第一列
    elif(type == 4):
        data = pd.read_table(filename, sep=',')
        x, y = data.ix[:, 1:], data.ix[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test

K_NN_type = 0
SKL = 'none'
train_X, predict_X, train_y, predict_y = loadData_split('E:/Sonar.txt', 3, 1, 1)
trainX = np.array(train_X)
predictX = np.array(predict_X)
trainy = np.array(train_y)
predicty = np.array(predict_y)
print(trainX, predictX, trainy, predicty)


num_fea_original=mat(trainX).shape[1]
feature=[]

for i in range(num_fea_original):
    feature.append(i)

#===========================================相关工具方法=============================
def read_data_fea(fea_list, dataset):
    dataMat = mat(dataset)
    col = dataMat.shape[0]#行号
    data_sample = []
    for i in range(col):
        col_i = []
        for j in fea_list:
            col_i.append(dataMat[i, j])
        data_sample.append(col_i)
    return data_sample

def numtofea(num, fea_list):
    feature = []
    for i in range(len(num)):
        if num[i] == 1:
            feature.append(fea_list[i])
        else:
            continue
    return feature

#======================================fitness function=================================

def get_fitness(data_train,label_train,data_pre,label_pre,train_cla):
    acc = 0
    if train_cla is 'train_knn':
        clf=neighbors.KNeighborsClassifier(n_neighbors=K_NN_type)#创建分类器对象
        if (len(data_train[0]) > 0):
            clf.fit(data_train, label_train)#用训练数据拟合分类器模型搜索
            predict=clf.predict(data_pre)
            acc=acc_pre(predict, label_pre)#预测标签和ground_true标签对比 算准确率
    elif train_cla is 'svm':
        clf = svm.SVC()
        if (len(data_train[0]) > 0):
            clf.fit(data_train, label_train)
            predict = clf.predict(data_pre)
            acc = acc_pre(predict, label_pre)
    return acc

def numtofea(num, fea_list):
    feature = []
    for i in range(len(num)):
        if num[i] == 1:
            feature.append(fea_list[i])
        else:
            continue
    return feature

def acc_pre(label_pre,label_train):
    num=0
    for i in range(len(label_pre)):
        if label_pre[i]!=label_train[i]:
            num+=1
    return (1-num/len(label_train))

def dr_pre(feature_list):
    feature_sum = len(feature_list)

    count = 0
    for i in feature_list:
        if(i == 1):
            count = count + 1
    return 1-(count/feature_sum)


def get_max(new_list):
    max = 0
    count = 0
    for i in new_list:
        count  = count + 1
        if(i > max):
            max = i
    return max, count - 1
#===================================初始化参数====================================

DNA_SIZE = len(trainX[0])         # parameter (solution) number
N_POP = 50           # population size
N_GENERATION = 1000   # training step
LR = 0.02

#==========================build multivariate distribution
mean = tf.Variable(tf.truncated_normal([DNA_SIZE, ], stddev=0.02, mean=0.5), dtype=tf.float32)
cov = tf.Variable(tf.eye(DNA_SIZE), dtype=tf.float32)
mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=abs(cov + tf.Variable(0.001 * tf.eye(DNA_SIZE), dtype=tf.float32)))
make_kid = mvn.sample(N_POP)

#==========================compute gradient and update mean and covariance matrix from sample and fitness
tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
loss = -tf.reduce_mean(mvn.log_prob(tfkids)*tfkids_fit + 0.01 * mvn.log_prob(tfkids) * mvn.prob(tfkids))         # log prob * fitness
# print(0.01 * mvn.log_prob(tfkids) * mvn.prob(tfkids))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss) # compute and apply gradients for mean and cov

sess = tf.Session()
sess.run(tf.global_variables_initializer())

max = 0
dr = 0
for g in range(N_GENERATION):
    if N_GENERATION % 10 == 0:
        LR = LR * 0.9
    kids = sess.run(make_kid)
    kids_fit = []
    feature_set = []
    for i in kids:
        feature_list = []
        k = 0
        for j in i:
            if j > 0.5:
                feature_list.append(1)
            else:
                feature_list.append(0)
        fea_list_CB = numtofea(feature_list, feature)
        feature_set.append(feature_list)
        data_sample = read_data_fea(fea_list_CB, trainX)
        data_predict = read_data_fea(fea_list_CB, predictX)
        kid_fit = get_fitness(data_sample, trainy, data_predict, predicty, SKL) * 100 - max
        kids_fit.append(kid_fit)
    sess.run(train_op, {tfkids_fit: kids_fit, tfkids: kids})  # update distribution parameters
    new_max, count = get_max(kids_fit)
    new_max = new_max + max
    feature_get = feature_set[count]
    if(new_max > max):
        max = new_max
        dr = dr_pre(feature_get)
        print('   使用', SKL, '第', g+1, '轮迭代：')
        print('选取特征为：', feature_get)
        print('维度缩减为：', dr)
        print('准确率为：', max)
        print(kids_fit)
    elif(new_max == max):
        new_dr = dr_pre(feature_get)
        if(new_dr > dr):
            dr = new_dr
            print('   使用', SKL,  '第', g+1, '轮迭代：')
            print('选取特征为：', feature_get)
            print('维度缩减为：', dr)
            print('准确率为：', max)
            print(kids_fit)
