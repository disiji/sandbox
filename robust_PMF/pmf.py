import tensorflow as tf


class PMF:
    """Build the graph for Probabilistic Matrix Factorization"""
    
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
        with tf.name_scope("data"):
            self.X_user = tf.placeholder(tf.int32, shape = [self.batch_size], name = "X_user")
            self.X_item = tf.placeholder(tf.int32, shape = [self.batch_size], name = "X_item")
            self.Y = tf.placeholder(tf.int32, shape = [self.batch_size], name = "Y")
    
    def _create_variables(self):
        """Step2: define embedding of users and items"""
        with tf.name_scope("embed"):
            self.U = tf.Variable(tf.random_uniform([self.num_users,self.rank],-1.0,1.0)
                                 ,name = "user_embed")
            self.V = tf.Variable(tf.random_uniform([self.num_items, self.rank],-1.0,1.0)
                                , name = "item_embed")
        self.var_list = [self.U, self.V]
            
    
    def _create_loss(self):
        """Step3: define the model and create the loss"""
        with tf.name_scope("loss"):
            # get innner product of user_embed and item_embed of input data
            self.X_user_embed = tf.nn.embedding_lookup(self.U, self.X_user, name = "X_user_embed")
            self.X_item_embed = tf.nn.embedding_lookup(self.V, self.X_item, name = "X_item_embed")
            self.pred = tf.reduce_sum(self.X_user_embed * self.X_item_embed,1)
            # define loss function under Gaussian assumption
            self.loss = tf.nn.l2_loss(self.pred - tf.to_float(self.Y)) \
                        + self.lambda_ * tf.nn.l2_loss(self.X_user_embed) \
                        + self.lambda_ * tf.nn.l2_loss(self.X_item_embed)
            
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
                            var_list = self.var_list, global_step=self.global_step)

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()