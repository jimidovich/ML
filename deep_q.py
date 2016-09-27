import sys
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt

from ml_fi import read_local_data
import add_feature as af
import performance as pf


def load_time_series(prod, prod_type='fut', freq='d1'):
    return read_local_data(prod, prod_type, freq).ix[:, :'close']


def make_time_series_states(df):
    '''price as state elements'''
    df = af.add_hist_price(df, n=[1, 2, 3, 4])
    df = af.add_hist_avg(df, 5, 10)
    df = af.add_hist_avg(df, 10, 20)
    df = af.add_hist_avg(df, 20, 40)
    df = af.add_hist_avg(df, 40, 80)
    df = af.add_hist_avg(df, 80, 160)
    df['equity'] = 100.0
    return df.ix[:, 'close':].dropna().values


def make_states(df):
    df = af.add_roc(df, n=[1, 2, 3, 4, 5])
    df = af.add_hist_avg(df, 5, 10)
    df = af.add_hist_avg(df, 10, 20)
    df = af.add_hist_avg(df, 20, 40)
    df = af.add_hist_avg(df, 40, 80)
    df = af.add_hist_avg(df, 80, 160)
    df.ix[:, -5:] = df.ix[:, -5:].apply(lambda x: x / df['close'])
    df['equity'] = 100.0
    return df.ix[:, 'roc_1':].dropna().values


def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.01, shape=shape), name=name)


def multilayer_nn(x, w, b):
    with tf.name_scope('Hidden_Layer_1'):
        layer1 = tf.add(tf.matmul(x, w['h1']), b['b1'])
        layer1 = tf.nn.relu(layer1)
    with tf.name_scope('Hidden_Layer_2'):
        layer2 = tf.add(tf.matmul(layer1, w['h2']), b['b2'])
        layer2 = tf.nn.relu(layer2)
    with tf.name_scope('Out_Layer'):
        out_layer = tf.matmul(layer2, w['out']) + b['out']
    tf.histogram_summary('output', out_layer)
    return out_layer


def train_nn(mode=None):
    prod = 'ag'
    prod_type = 'fut'

    df = load_time_series(prod, prod_type)
    states = make_states(df)
    # states = states[-400:, :]
    ret = pd.Series(states[0, :]).pct_change().shift(-1)
    n_train = int(states.shape[0] * 0.9)
    # n_test = states.shape[0] - n_train

    n_input = states.shape[1]
    n_h1 = 64
    n_h2 = 50
    n_classes = 3
    # learning_rate = 0.001
    n_observe = 80
    batch_size = 40
    gamma = 0.8
    if mode == 'test':
        epochs = 1
        epsilon = 0.0
    else:
        epochs = 10000
        epsilon = 1.0

    with tf.name_scope('Training_Data'):
        x = tf.placeholder('float', [None, n_input], name='x')
        a = tf.placeholder('float', [None, n_classes], name='a')
        y = tf.placeholder('float', [None], name='y')
    w = {'h1': weight_variable([n_input, n_h1], name='W_h1'),
         'h2': weight_variable([n_h1, n_h2], name='W_h2'),
         'out': weight_variable([n_h2, n_classes], name='W_out')}
    b = {'b1': bias_variable([n_h1], name='b_h1'),
         'b2': bias_variable([n_h2], name='b_h2'),
         'out': bias_variable([n_classes], name='b_out')}
    pred = multilayer_nn(x, w, b)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    pred_action = tf.reduce_sum(tf.mul(pred, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - pred_action))
    tf.scalar_summary('cost', cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.initialize_all_variables()

    display_step = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, 'saved_dqmlp')
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('logs/', sess.graph)
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded:', checkpoint.model_checkpoint_path)
        else:
            print('Could not find old network weights')
        plt.figure()
        # plt.ion()
        # plt.show()
        for epoch in range(epochs):
            start_time = time.time()
            states[:, -1] = 100
            avg_cost = 0.0
            alist = []
            reward = []
            for t in range(n_train - 1):
                q = pred.eval(feed_dict={x: [states[t, :]]})[0]
                a_t = np.zeros([n_classes])
                if (random.random() < epsilon):
                    a_idx = np.random.choice(n_classes)
                else:
                    a_idx = np.argmax(q)
                a_t[a_idx] = 1

                alist.append(a_t)
                ret = states[t+1, 0]  # states[0] is roc_1
                agent_ret = ret * (a_idx - 1)
                states[t+1, -1] = states[t, -1] * (1 + agent_ret)
                sharpe = pf.sharpe(pd.Series(states[t-30:t+2, -1]))\
                    if t > 30 else 0
                reward.append(agent_ret + gamma * sharpe)

                if t > n_observe:
                    minibatch = random.sample(range(t-n_observe, t+1),
                                              batch_size)
                    next_t = [i+1 for i in minibatch]
                    next_q = pred.eval(feed_dict={x: states[next_t, :]})
                    y_batch = [reward[i] for i in minibatch] +\
                        gamma*np.max(next_q, axis=1)
                    _, c = sess.run([optimizer, cost], feed_dict={
                        x: states[minibatch, :],
                        a: [alist[i] for i in minibatch],
                        y: y_batch})
                    avg_cost += c / (t + 1)
                    if t == n_train-2:
                        result = sess.run(merged, feed_dict={
                            x: states[minibatch, :],
                            a: [alist[i] for i in minibatch],
                            y: y_batch})
                        writer.add_summary(result, epoch*n_train + t)

            sharpe = pf.sharpe(pd.Series(states[:n_train, -1]))
            # print(sharpe)
            # print(alist)
            # plt.ion()
            if mode == 'test':
                plt.plot(states[:n_train, -1])
                # plt.pause(0.0001)
                df.close[160:160+n_train].reset_index(drop=True).plot(
                    secondary_y=True)
                plt.show()
            end_time = time.time()
            if epoch % display_step == 0:
                print('Epoch {}, cost={:.6f}, equity={:.2f},\
                      sh={:.4f}, time={}'.format(
                          epoch, avg_cost, states[n_train-1, -1], sharpe,
                          end_time-start_time))
            if epoch % 100 == 1:
                saver.save(sess, 'saved_networks/' + prod + '-dqn',
                           global_step=epoch*n_train)
            if epsilon > 0.1:
                epsilon -= 1/epochs

        print('Training finished.')


if __name__ == '__main__':
    sys.exit(train_nn(*sys.argv[1:]))
