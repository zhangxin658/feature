import numpy as np
from numpy import *#mat
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd


#首先是定义全局变量
DNA_BOUND = [0, 5]       # solution upper and lower bounds//参数选取范围
N_GENERATIONS = 200      # 迭代数量
POP_SIZE = 100           # population size//种族数量
N_KID = 50               # n kids per generation //每一代有n个孩子


#以下是读取文件中的参数
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

#以下是保存数据
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

K_NN_type = 0
SKL = 'none'
train_X, predict_X, train_y, predict_y = loadData_split('E:/Vehicle.txt', 1, 5, 1)
trainX = np.array(train_X)
predictX = np.array(predict_X)
trainy = np.array(train_y)
predicty = np.array(predict_y)
DNA_SIZE = len(trainX[0])
saveData1('E:/trainX.txt', trainX)
saveData1('E:/predictX.txt', predictX)
saveData2('E:/trainy.txt', trainy)
saveData2('E:/predicty.txt', predicty)
print(trainX, predictX, trainy, predicty)

def make_kid(pop, n_kid):  #传入的参数包括父代节点和要生成孩子的个数
    # generate empty kid holder //
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}#生成孩子参数
    kids['mut_strength'] = np.empty_like(kids['DNA']) #生成孩子的变异强敌
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):  #在
        # crossover (roughly half p1 and half p2)
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)#产生随机采样,从种群中找到两个，并且不能重复
        print('p1, p2:', p1, p2)
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        print('cp:', cp)
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids

def kill_bad(pop, kids):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop
# find non-zero fitness for selection
def get_fitness(pred): return pred.flatten()    #返回适应度值

def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function//

pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values //初始化参数；
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))
#以下是网络中的主循环
for _ in range(N_GENERATIONS):

    kids = make_kid(pop, N_KID)  # 通过父代生成子代
    pop = kill_bad(pop, kids)   # keep some good parent for elitism