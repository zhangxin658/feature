from numpy import *#mat
from sklearn import neighbors
from sklearn import svm
from sklearn import tree


def loadData3(filename):
    numFeat=len(open(filename).readline().split(' '))-1
    feature=[];label=[]
    fr=open(filename)
    for line in fr.readlines():
        xi=[]
        curline=line.strip().split(' ')
        for i in range(numFeat):
            xi.append(float(curline[i]))
        feature.append(xi)
        label.append((curline[-1]))
    return feature, label
def loadData1(filename):
    numFeat = len(open(filename).readline().split(' '))
    data = []
    fr = open(filename)
    for line in fr.readlines():
        xi=[]
        curline=line.strip().split(' ')
        for i in range(numFeat):
            xi.append(float(curline[i]))
        data.append(xi)
    return data
def loadData2(filename):
    numFeat = len(open(filename).readline().split(' ')) - 1
    data = []
    feature = []
    fr = open(filename)
    for line in fr.readlines():
        xi=[]
        curline=line.strip().split(' ')
        for i in range(numFeat):
            xi.append(float(curline[i]))
        feature.append(xi)
        data.append((curline[-1]))
    return data
train_X, train_y = loadData3('E:/Vehicle.txt')#trainX,trainy are all list
predict_X, predict_y = loadData3('E:/Vehicle.txt')
trainX = loadData1('E:/dataset/trainX.txt')
trainy = loadData2('E:/dataset/trainy.txt')
predictX = loadData1('E:/dataset/predictX.txt')
predicty = loadData2('E:/dataset/predicty.txt')

print(mat(trainX).shape[1])
print(mat(trainy).shape[1])

# print(train_X)
# print(train_y)

print(trainX)
print(trainy)
num_fea_original = mat(trainX).shape[1]
feature = []


for i in range(num_fea_original):
    feature.append(i)             #得到特征列表，比如对于18维特征[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

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
def get_fitness(data_train,label_train,data_pre,label_pre,train_cla):
    acc = 0
    if train_cla is 'train_knn':
        print('使用knn')
        clf=neighbors.KNeighborsClassifier(n_neighbors=5)#创建分类器对象
        if (len(data_train[0]) > 0):
            clf.fit(data_train, label_train)#用训练数据拟合分类器模型搜索
            predict=clf.predict(data_pre)
            acc=acc_pre(predict, label_pre)#预测标签和ground_true标签对比 算准确率
    elif train_cla is 'svm':
        print('使用svm')
        clf = svm.SVC()
        clf.fit(data_train, label_train)
        predict = clf.predict(data_pre)
        acc = acc_pre(predict, label_pre)
        return acc
    elif train_cla is 'tree':
        print('使用tree')
        clf = tree.DecisionTreeClassifier()
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
    return feature                          #得到所选取的特征所在的位置，比如[5, 8]，则说明所选的特征在第六个和第九个位置

def acc_pre(label_pre,label_train):
    num=0
    print('len(predict)', len(label_pre))
    for i in range(len(label_pre)):
        if label_pre[i]!=label_train[i]:
            num+=1
    return (1-num/len(label_train))

feature_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
fea_list_CB = numtofea(feature_list, feature)            #得到所选取的特征所在的位置，比如[5, 8]，则说明所选的特征在第六个和第九个位置

data_sample = read_data_fea(fea_list_CB, trainX)             #得到给位置所对应的真实的特征值
print(data_sample)
data_predict = read_data_fea(fea_list_CB, predictX)
print(data_predict)
kid_fit = get_fitness(data_sample, trainy, data_predict, predicty, 'train_knn') * 100
print(kid_fit)
