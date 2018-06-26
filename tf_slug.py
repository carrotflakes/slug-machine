from slug_machine import SlugMachine, Slug
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as core_layers


class TFSlug(Slug):

    def __init__(self, tape_width=32, state_size=32):
        self.tape_width = tape_width
        self.state_size = state_size
        self.state = np.zeros((self.state_size,), dtype=np.float32)
        self.tape_pl = tf.placeholder(tf.int8, shape=(None, None, self.tape_width))
        self.state_pl = tf.placeholder(tf.float32, shape=(None, self.state_size))
        self.next_tape_pl = tf.placeholder(tf.int8, shape=(None, None, self.tape_width))

        batch_size = tf.shape(self.tape_pl)[0]
        seq_len = tf.shape(self.tape_pl)[1]

        cell = tf.contrib.rnn.GRUCell(
            self.state_size,
            activation=tf.nn.elu,
            kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=0),
            bias_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=0))
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell,
            self.tape_width)

        tape = tf.cast(self.tape_pl, tf.float32)

        next_tape_raw, self.next_state = tf.nn.dynamic_rnn(
            cell,
            tape,
            initial_state=self.state_pl)

        self.next_tape = tf.sign(next_tape_raw) > 0

        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.cast(self.next_tape_pl, tf.float32),
            logits=next_tape_raw)

        optimizer = tf.train.AdamOptimizer(0.01)
        self.optimize = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, chunk):
        tape, state = self.sess.run([self.next_tape, self.next_state], {
            self.tape_pl: np.array([[chunk]]),
            self.state_pl: np.array([self.state])
        })
        self.state = state[0]
        return tape[0][0]

    def learn(self, slug_trace, window_size=128, epoch=100):
        episodes = slug_trace.get_tape_transitions()
        batch_size = len(episodes)
        batches = []
        while sum(map(len, episodes)) > 0:
            tape_before, tape_after = [], []
            size = min(window_size, max(map(len, episodes)))
            for episode in episodes:
                pairs = episode[:size]
                pairs += [(np.zeros((self.tape_width,), dtype=np.bool),) * 2] * (size - len(pairs))
                episode[:size] = []
                t1, t2 = zip(*pairs)
                tape_before.append(t1)
                tape_after.append(t2)
            batches.append((tape_before, tape_after))

        for i in range(epoch):
            state = [np.zeros((self.state_size,), dtype=np.float32)] * batch_size
            total_loss = 0
            for tape_before, tape_after in batches:
                loss, _, state = self.sess.run([self.loss, self.optimize, self.next_state], {
                    self.tape_pl: tape_before,
                    self.next_tape_pl: tape_after,
                    self.state_pl: state
                })
                total_loss += loss
            print(total_loss)


if __name__ == '__main__':
    slug = TFSlug()
    chunk = np.zeros((32,))
    slug.step(chunk)
