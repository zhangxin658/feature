import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

scope = "zhangxin"

mean1 = tf.Variable(tf.truncated_normal([5, ], stddev=0.1, mean=2,), dtype=tf.float32, name='mean1')
cov1 = tf.Variable(tf.eye(5), dtype=tf.float32, name='cov1')

mvn1 = MultivariateNormalFullCovariance(loc=mean1, covariance_matrix=cov1, name=scope + '_mvn1')
kids1 = mvn1.sample(5)

mean2 = tf.Variable(tf.truncated_normal([5, ], stddev=0.1, mean=20), dtype=tf.float32, name='mean2')
cov2 = tf.Variable(tf.eye(5), dtype=tf.float32, name='cov2')
mvn2 = MultivariateNormalFullCovariance(loc=mean2, covariance_matrix=cov2, name='mvn2', )
kids2 = mvn2.sample(5)
mvn2 = mvn1.copy()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(kids1))
print('====================================================')
print(sess.run(kids2))
print('====================================================')
print(mvn1.mean)
print(mvn2.mean)
print('====================================================')
print(sess.run(mvn1.mean()))
print(sess.run(mvn2.mean()))
print('====================================================')
print(sess.run(mvn1.sample(5)))
print('====================================================')
print(sess.run(mvn2.sample(5)))
print(mvn1.name)