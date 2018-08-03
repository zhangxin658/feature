"""
The basic idea about Nature Evolution Strategy with visualation.

Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/

Dependencies:
Tensorflow >= r1.2
numpy
matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

DNA_SIZE = 2         # parameter (solution) number
N_POP = 20           # population size
N_GENERATION = 1000   # training step
LR = 0.02           # learning rate
max_fit = 0


# fitness function    np.sin(10*x)*x + np.cos(2*x)*x
def get_fitness(pred): return - pred[:, 0]**2 - pred[:, 1]**2
def get_minus_fit(fits):
    fitness = []
    for i in fits:
        fitness.append(i - max_fit)
    return fitness

def get_max_fit(fits):
    global max_fit
    for i in fits:
        if i > max_fit:
            max_fit = i

# build multivariate distribution
mean = tf.Variable(tf.random_normal([2, ], 13., 0.1), dtype=tf.float32)
cov = tf.Variable(5. * tf.eye(DNA_SIZE), dtype=tf.float32)
mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=abs(cov))
make_kid = mvn.sample(N_POP)                                    # sampling operation

# compute gradient and update mean and covariance matrix from sample and fitness
tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
loss = -tf.reduce_mean(mvn.log_prob(tfkids)*tfkids_fit * 1.2)         # log prob * fitness
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss) # compute and apply gradients for mean and cov

sess = tf.Session()
sess.run(tf.global_variables_initializer())                     # initialize tf variables

# something about plotting (can be ignored)
n = 300
x = np.linspace(-20, 20, n)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        Z[i, j] = get_fitness(np.array([[x[i], x[j]]]))
plt.contourf(X, Y, -Z, 100, cmap=plt.cm.rainbow); plt.ylim(-20, 20); plt.xlim(-20, 20); plt.ion()

# training
for g in range(N_GENERATION):
    # if g % 10 == 0:
    #     LR = LR * pow(0.9, g / 10)
    #     print(g+1, 'LR', LR)
    kids = sess.run(make_kid)
    print(sess.run(mvn.mean()))
    print(sess.run(mvn.covariance()))
    kids_fit = get_fitness(kids)
    get_max_fit(kids_fit)
    kids_fits = get_minus_fit(kids_fit)
    sess.run(train_op, {tfkids_fit: kids_fits, tfkids: kids})    # update distribution parameters
    # plotting update
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(kids[:, 0], kids[:, 1], s=30, c='k');plt.pause(0.01)

print('Finished'); plt.ioff(); plt.show()