import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import scipy.io
import numpy as np
from robust_MF import RobustGaussianMF, RobustPoissonMF
from pmf import PMF
import tensorflow as tf

DISTR_DICT = ["pmf","robust_gaussian","robust_poisson"]
DATA_DICT = ["movielens","tw_oc"]

model_type = "robust_gaussian"
dataset = "movielens"

if dataset == "movielens":
    datafile = "movielens/"
    train_data = scipy.io.loadmat(datafile+'moviedata.mat')['train_vec'] # Y takes value from {1,2,3,4,5}
    train_data = np.array(train_data).astype(np.int32)
    train_data[:,0:2] -= 1

    test_data = scipy.io.loadmat(datafile+'moviedata.mat')['probe_vec']
    test_data = np.array(test_data).astype(np.int32)
    test_data[:,0:2] -= 1

    num_users = 6040
    num_items = 3952

if dataset == "tw_oc":
    datafile = "tw_oc/"
    train_data = np.genfromtxt(datafile + 'train_val.csv', delimiter=',').astype(np.int32)
    test_data = np.genfromtxt(datafile + 'test.csv', delimiter=',').astype(np.int32)

    num_users = 13559
    num_items = 11347

rank = 30
batch_size = 1000 
learning_rate = .001
lambda_ = .001 #regularization parameter
n_epochs = 500
N = train_data.shape[0] #900000
n_batches = N / batch_size

def main():
# build the model
    print "Rank of embedding is %d" % rank
    if model_type == "pmf":
        print "Building Graph for Probabilisic MF"
        model = PMF(num_users, num_items, rank, batch_size, learning_rate, lambda_)
    if model_type == "robust_gaussian":
        print "Building Graph for Robust Gaussian MF"
        model = RobustGaussianMF(num_users, num_items, rank, batch_size, learning_rate, lambda_)
    if model_type == "robust_poisson":
        print "Building Graph for Robust Poisson MF"
        model = RobustPoissonMF(num_users, num_items, rank, batch_size, learning_rate, lambda_)
    model.build_graph()
    print "Model Built!"

    with tf.Session() as sess:
# train the model
        print "Optimizing the Model"
        sess.run(tf.global_variables_initializer())
        for epoch_idx in xrange(n_epochs):
            loss_tracker = 0.
            for batch_idx in xrange(n_batches): 
                X_user_batch = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size,0]
                X_item_batch = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size,1]
                Y_batch = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size,2]
                # perform update
                loss_batch, _ = sess.run([model.loss, model.optimizer], 
                                      feed_dict={model.X_user: X_user_batch,
                                                 model.X_item: X_item_batch, 
                                                 model.Y: Y_batch})
                loss_tracker += loss_batch
            if (epoch_idx+1) % 10 == 0:
                print "Epoch %d. Obj: %.3f" % (epoch_idx+1, loss_tracker)
                print "Epoch %d. Beta: %.3f" % (epoch_idx+1, model.beta.eval())
                if model_type == "robust_gaussian":
                    print "Epoch %d. std: %.3f" % (epoch_idx+1, model.sigma.eval())
        print "Optimizaiton Finished!"

        np.savetxt(datafile+"U_dim_%d.csv" % rank, model.U.eval(), delimiter =',')
        np.savetxt(datafile+"V_dim_%d.csv" % rank, model.V.eval(), delimiter =',')

# test the model
        test_X_user = tf.convert_to_tensor(test_data[:,0])
        test_X_item = tf.convert_to_tensor(test_data[:,1])
        test_Y = tf.to_float(tf.convert_to_tensor(test_data[:,2])) 

        if model_type == "pmf" or "robust_gaussian":
            test_user_embed = tf.nn.embedding_lookup(model.U, test_X_user, name = "X_user_embed")
            test_item_embed = tf.nn.embedding_lookup(model.V, test_X_item, name = "X_item_embed")
            test_pred = tf.reduce_sum(test_user_embed * test_item_embed,1)   
            # clip prediction to 1 - 5 for movielens predicton 
            test_pred = tf.clip_by_value(test_pred,1,5)
            test_error = tf.reduce_mean(tf.squared_difference(test_pred, test_Y))
            print "Avg. Square Error per Point of Test Data is %.3f" % test_error.eval()

        if model_type == "robust_poisson":
            # compute avg. log likelihood under categorical distribution
            score_mat = tf.exp(tf.matmul(model.U,tf.transpose(model.V)))
            score_mat /= tf.expand_dims(tf.reduce_sum(score_mat, 1),1)
            index = tf.transpose(tf.concat([[test_X_user], [test_X_item]], 0))
            log_catogorical = tf.reduce_sum(test_Y * tf.log(tf.gather_nd(score_mat,index))) / tf.reduce_sum(test_Y)
            print "Avg. Log Catogerical Likelihood per Point on Test Data is %.3f" % log_catogorical.eval()

            # compute avg. log likelihood under Poisson distribution
            test_user_embed = tf.nn.embedding_lookup(model.U, test_X_user, name = "X_user_embed")
            test_item_embed = tf.nn.embedding_lookup(model.V, test_X_item, name = "X_item_embed")
            test_pred = tf.reduce_sum(test_user_embed * test_item_embed,1) 
            test_Poissons = tf.contrib.distributions.Poisson(rate = tf.exp(test_pred))
            log_poisson = tf.reduce_mean(test_Poissons.log_prob(test_Y))
            print "Avg. Log Poisson Likelihood per Point on Test Data is %.3f" % log_poisson.eval()

# write model and summaries to file: TO DO
if __name__ == '__main__':
    main()