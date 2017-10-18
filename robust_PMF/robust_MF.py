import tensorflow as tf
from scipy.stats import poisson
import numpy as np

class RobustMF(object):

    def __init__(self, num_users, num_items, rank, batch_size,learning_rate,lambda_):
        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def _create_placeholders(self):
        """Step1: define place holders for input and output"""
        print "creating placeholders"
        with tf.name_scope("data"):
            self.X_user = tf.placeholder(tf.int32, shape=[self.batch_size], name="X_user")
            self.X_item = tf.placeholder(tf.int32, shape=[self.batch_size], name="X_item")
            self.Y = tf.to_float(tf.placeholder(tf.int32, shape=[self.batch_size], name="Y"))

    def _create_robust_embedding(self):
        """Step2: define embedding of users and items"""
        print "creating embeddings"
        with tf.name_scope("embed"):
            self.U = tf.Variable(tf.random_uniform([self.num_users,self.rank], -1.0, 1.0)
                                 , name="user_embed")
            self.V = tf.Variable(tf.random_uniform([self.num_items, self.rank], -1.0, 1.0)
                                 , name="item_embed")
        with tf.name_scope("beta"):
            self.beta = tf.Variable(np.array([0.01]), name="global_beta", dtype=tf.float32)
            # define Beta distribution prior on self.beta, then self.beta takes value between 0 and 1
            # then beta divergence lies between MLE and l2E
            self.prior_alpha, self.prior_beta = 20, 20
        with tf.name_scope("predict"):
            self.X_user_embed = tf.nn.embedding_lookup(self.U, self.X_user, name = "X_user_embed")
            self.X_item_embed = tf.nn.embedding_lookup(self.V, self.X_item, name = "X_item_embed")
            self.pred = tf.reduce_sum(self.X_user_embed * self.X_item_embed,1)
    
    def _create_variables(self):
        """Step2: define other variables"""
        raise NotImplementedError()
    
    def _create_loss(self):
        """Step3: define the model and create the loss"""
        raise NotImplementedError()
            
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        print "creating optimizer"
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(-self.loss,
                            var_list = self.var_list, global_step=self.global_step)

    def build_graph(self):
        self._create_placeholders()
        self._create_robust_embedding()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()


class RobustGaussianMF(RobustMF):
    
    def _create_variables(self):
        print "creating variables"
        with tf.name_scope("dist"):
            self.sigma = tf.Variable(np.array([1]), name = "gaussian_std", dtype = tf.float32)
        self.var_list = [self.U, self.V, self.beta, self.sigma]
#        self.var_list = [self.U, self.V, self.beta]
    
    def _create_loss(self):
        print "creating loss function"
        with tf.name_scope("loss"):
            # self.loss = tf.reduce_sum(tf.exp(self.beta / (-2 * self.sigma * self.sigma) \
            #           * tf.square(self.pred - self.Y)))
            # self.loss -= self.beta * (self.beta+1) ** (-3. / 2)
            # self.loss *= (2*3.1415*self.sigma*self.sigma) ** (-self.beta / 2)
            # self.loss -= self.lambda_ * tf.nn.l2_loss(self.X_user_embed) \
            #             + self.lambda_ * tf.nn.l2_loss(self.X_item_embed)
            # self.loss += (self.prior_alpha - 1) * tf.log(self.beta) \
            #             + (self.prior_beta - 1) * tf.log(1-self.beta)

            self.loss = tf.reduce_sum(tf.square(self.pred - self.Y)/(-2 * self.sigma * self.sigma))
            self.loss -= self.beta * (self.beta+1) ** (-3. / 2)*(2*3.1415*self.sigma*self.sigma) ** (-self.beta / 2)
            self.loss -= self.lambda_ * tf.nn.l2_loss(self.X_user_embed) \
                        + self.lambda_ * tf.nn.l2_loss(self.X_item_embed)
            self.loss += (self.prior_alpha - 1) * tf.log(self.beta) \
                        + (self.prior_beta - 1) * tf.log(1-self.beta)


class RobustPoissonMF(RobustMF):
    
    def _create_variables(self):
        print "creating variables"
        self.var_list = [self.U, self.V, self.beta]

    def _create_loss(self):
        print "creating loss function"
        with tf.name_scope("loss"):
            Poissons = tf.contrib.distributions.Poisson(rate = tf.exp(self.pred), name = "poisson_per_datapoint")
            self.loss = tf.reduce_sum(tf.exp(self.beta * Poissons.log_prob(self.Y)))

            # self.power_ingeral = tf.zeros_like(self.pred)
            # print "evaluating Poison power integral"
            # for _ in range(- self.Poisson_bandwidth, self.Poisson_bandwidth):
            #     if _ % 100 == 0:
            #         print _
            #     self.power_ingeral += tf.exp((self.beta + 1) * Poissons.log_prob(_ + tf.exp(self.pred)))    

            # self.loss -= self.beta / (self.beta + 1) * tf.reduce_sum(self.power_ingeral)

            self.loss -= tf.reduce_sum(self.beta * (self.beta + 1) ** (- 3./2) * \
                        (2 * 3.1415 * tf.exp(self.pred)) ** (- self.beta / 2))
            self.loss -= self.lambda_ * tf.nn.l2_loss(self.X_user_embed) \
                        + self.lambda_ * tf.nn.l2_loss(self.X_item_embed)
            self.loss += (self.prior_alpha - 1) * tf.log(self.beta) \
                        + (self.prior_beta - 1) * tf.log(1-self.beta)