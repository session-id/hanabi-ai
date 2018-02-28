from replay_buffer import ReplayBuffer
import os
import time
from collections import deque
import numpy as np
import tensorflow as tf


class RL_Model(object):
    def __init__(self, inputs, config, train_simulator, test_simulator):
        self.inputs = inputs
        self.config = config
        self.train_simulator = train_simulator
        self.test_simulator = test_simulator

    def get_action(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class QL_Model(RL_Model):
    '''
    Abstract Q-learning model.

    Subclasses must implement the _get_q_values_op() method.
    '''

    def __init__(self, inputs, config, train_simulator, test_simulator):
        super(RL_Model, self).__init__(
            inputs=inputs,
            config=config,
            train_simulator=train_simulator,
            test_simulator=test_simulator
        )

        # create dirs for summaries + saved weights if needed
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)

        if config.gpu > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)
        else:
            self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)

        # build the graph
        self.build()


    def _get_q_values_wrapper(self, state, scope):
        '''
        Args
        - state: tf.Tensor, shape [batch_size, state_dim]
        - scope: str, name of scope

        Returns
        - q: tf.Tensor, shape [batch_size, num_actions], with any invalid actions having a q-value of -inf
        '''
        q_values = self._get_q_values_op(state, scope)
        neg_inf = tf.constant(-np.inf, dtype=tf.float32, shape=q_values.shape)
        q = tf.where(self.placeholders['valid_actions_mask'], q_values, neg_inf)
        return q


    def _get_q_values_op(self, state, scope):
        '''
        Args
        - state: tf.Tensor, shape [batch_size, state_dim]
        - scope: str, name of scope

        Returns
        - q: tf.Tensor, shape [batch_size, num_actions]
        '''
        raise NotImplementedError


    def save(self):
        '''
        Saves the weights of the current model to config.ckpt_dir with prefix 'ckpt'
        '''
        ckpt_prefix = os.path.join(self.config.ckpt_dir, 'ckpt')
        self.saver.save(self.sess, ckpt_prefix)


    def get_action(self, state):
        '''
        Performs epsilon-greedy action selection.

        TODO: this currently uses config.soft_epsilon, but we should use a decreasing
            epsilon over time.

        Args
        - state: np.array, shape [state_dim]

        Returns: (action, q_values)
        - action: int, index of the action to take
        - q_values: np.array, type float32, vector of Q-values for the given state
        '''
        q_values = self.sess.run(self.q, feed_dict={self.placeholders['states']: [state]})[0]
        if np.random.random() < self.config.soft_epsilon:
            valid_actions = list(self.train_simulator.get_valid_actions(state))
            random_action = np.random.choice(valid_actions)
            return random_action, q_values
        else:
            return np.argmax(q_values), q_values


    def update_averages(self, split, ep_reward=None, q_values=None):
        '''
        Args
        - ep_reward: float, the total reward for an episode
        - q_values: list of float, the q_values from a single step
        '''
        if split not in ['train', 'test']:
            raise ValueError("unrecognized split")

        metrics_dict = self.metrics_train if split == 'train' else self.metrics_test
        if ep_reward is not None:
            metrics_dict.append(ep_reward)
        if q_values is not None:
            metrics_dict['q_values'] += q_values


    def train(self):
        '''Train the model

        Returns
        - test_avg_rewards: list of float, avg rewards from self.evaluate() every config.test_freq steps
        '''
        test_avg_rewards = []

        # initialize replay buffer
        replay_buffer = ReplayBuffer() # TODO
        num_actions = self.train_simulator.get_num_actions()

        episode = 0
        step = 0

        # train for self.config.train_num_steps steps
        while step < self.config.train_num_steps:
            total_reward = 0
            state = self.train_simulator.get_start_state()

            # train multiple episodes
            while True:
                start_time = time.time()

                action, q_values = self.get_action(state)
                new_state, reward, done = self.train_simulator.take_action(state, action)

                valid_actions_mask = np.zeros(num_actions, dtype=bool)
                valid_action_indices = list(self.train_simulator.get_valid_actions(state))
                valid_actions_mask[valid_action_indices] = True

                replay_buffer.store(step, state, valid_actions_mask, action, reward, done, new_state)

                state = new_state
                total_reward += reward
                self.update_averages('train', reward=None, q_values=q_values)

                if step > self.config.train_start:
                    if step % self.config.test_freq == 0:
                        test_rewards = self.evaluate(step=step)
                        test_avg_rewards.append(np.mean(test_rewards))

                    if step % self.config.print_freq == 0:
                        loss = self.train_step(step, replay_buffer, return_stats=True)
                        duration = time.time() - start_time
                        print('Step {:05d}. Episode {:02d}. loss: {:0.4f}, time: {:0.3f}s'.format(
                            step, episode, loss, duration))
                    else:
                        self.train_step(step, replay_buffer, return_stats=False)

                step += 1
                if done or step >= self.config.train_num_steps:
                    break

            episode += 1
            self.update_averages('train', reward=total_reward, q_values=None)

        # evaluate again at the end of training
        test_avg_rewards.append(self.evaluate(step=step))
        return test_avg_rewards


    def evaluate(self, step=None):
        '''Evaluate model

        Args
        - step: int, current training step
            or None to avoid saving summaries to TensorBoard

        Returns
        - rewards: list of float, rewards on config.test_num_episodes
        '''
        num_episodes = self.config.test_num_episodes
        rewards = np.zeros(num_episodes, dtype=np.float64)

        for ep in range(self.config.test_num_episodes):
            total_reward = 0
            state = self.test_simulator.get_start_state()

            while True:
                action, q_values = self.get_action(state)
                state, reward, done = self.test_simulator.take_action(state, action)
                if step is not None:
                    self.update_averages('test', reward=None, q_values=q_values)
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards[ep] = total_reward

            if step is not None:
                self.update_averages('test', reward=total_reward, q_values=None)

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, std_reward)
        print(msg)

        if step is not None:
            # write the summary to TensorBoard
            summary_fd = {
                self.summary_placeholders['rewards']: self.metrics_test['rewards'],
                self.summary_placeholders['q_values']: self.metrics_test['q_values']
            }
            test_summary_str = self.sess.run(self.summaries_test, feed_dict=summary_fd)
            self.summary_writer.add_summary(test_summary_str, step)
            self.summary_writer.flush()

        return rewards


    def train_step(self, step, replay_buffer, return_stats=False):
        '''
        Run a training step

        Args
        - step: int
        - replay_buffer: ReplayBuffer,
        - return_stats: bool

        Returns
        - if return_stats=True, returns loss: float
        - otherwise, does not return anything
        '''
        batch = replay_buffer.sample(self.config.batch_size)

        feed_dict = {
            self.placeholders['states']:             batch['states'],
            self.placeholders['actions']:            batch['actions'],
            self.placeholders['rewards']:            batch['rewards'],
            self.placeholders['states_next']:        batch['states_next'],
            self.placeholders['valid_actions_mask']: batch['valid_actions_mask'],
            self.placeholders['done_mask']:          batch['done_mask'],
            self.placeholders['lr']: self.config.lr
        }

        if return_stats:
            _, loss  = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

            # write the summary to TensorBoard
            summary_fd = {
                self.summary_placeholders['rewards']: self.metrics['rewards'],
                self.summary_placeholders['q_values']: self.metrics['q_values']
            }
            train_summary_str = self.sess.run(self.summaries_train, feed_dict=summary_fd)
            self.summary_writer.add_summary(train_summary_str, step)
            self.summary_writer.flush()
            return loss

        else:
            self.sess.run(self.train_op, feed_dict=feed_dict)

        # occasionaly update target network with q network
        if step % self.config.target_update_freq == 0:
            self.sess.run(self.update_target_op)


    def build(self):
        # self.placeholders: dict, {str => tf.placeholder}
        self._add_placeholders()

        # self.loss
        self._add_loss_op()

        # self.update_target_op
        self._add_update_target_op()

        # self.summary_placeholders, self.summaries_train, self.summaries_test
        self._add_summaries()

        # compute Q values
        self.q = self._get_q_values_wrapper(self.placeholders['states'], scope="q")
        self.target_q = self._get_q_values_wrapper(self.placeholders['states_next'], scope="target_q")

        # self.train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])
        self.train_op = optimizer.minimize(self.loss)


    def _add_summaries(self):
        '''
        Creates summary ops. Defines 5 new properties on self:
        - self.metrics_train
        - self.metrics_test
        - self.summary_placeholders
        - self.summaries_train
        - self.summaries_test
        '''
        self.metrics_train = {
            'rewards': deque(maxlen=self.config.num_episodes_test),
            'q_values': deque(maxlen=self.config.q_values_metrics_size)
        }
        self.metrics_test = {
            'rewards': deque(maxlen=self.config.num_episodes_test),
            'q_values': deque(maxlen=self.config.q_values_metrics_size)
        }
        self.summary_placeholders = {
            'rewards': tf.placeholder(tf.float32, shape=[None], name='rewards'),
            'q_values': tf.placeholder(tf.float32, shape=[None], name='q_values')
        }

        avg_reward, var_reward = tf.nn.moments(self.summary_placeholders['rewards'], axes=[0])
        avg_q, var_q = tf.nn.moments(self.summary_placeholders['qs'], axes=[0])

        summary_tensors = {
            'avg_reward': avg_reward,
            'max_reward': tf.reduce_max(self.summary_placeholders['rewards']),
            'std_reward': tf.sqrt(var_reward),
            'avg_q': avg_q,
            'max_q': tf.reduce_max(self.summary_placeholders['q_values']),
            'std_q': tf.sqrt(var_q)
        }

        with tf.variable_scope('train'):
            self.summaries_train = tf.summary.merge([
                tf.summary.scalar(name, summary_tensor)
                for name, summary_tensor in summary_tensors.items()
            ] + [tf.summary.scalar("loss", self.loss)])

        with tf.variable_scope('test'):
            self.summaries_test = tf.summary.merge([
                tf.summary.scalar(name, summary_tensor)
                for name, summary_tensor in summary_tensors.items()
            ])


    def _add_placeholders(self):
        state_dim = self.train_simulator.get_state_vector_size()
        num_actions = self.train_simulator.get_num_actions()
        self.placeholders = {
            'states'      : tf.placeholder(tf.float32, shape=[None, state_dim]),
            'actions'     : tf.placeholder(tf.int32,   shape=[None]),
            'rewards'     : tf.placeholder(tf.float32, shape=[None]),
            'states_next' : tf.placeholder(tf.float32, shape=[None, state_dim]),
            'lr'          : tf.placeholder(tf.float32, shape=[]),
            'done_mask'   : tf.placeholder(tf.bool,    shape=[None]),

            # boolean mask of valid actions given the current batch of states
            'valid_actions_mask' : tf.placeholder(tf.bool, shape=[None, num_actions])
        }


    def _add_loss_op(self):
        '''
        Sets self.loss to the loss operation defined as

        Q_samp(s) = r if done
                  = r + gamma * max_a' Q_target(s', a')
        loss = (Q_samp(s) - Q(s, a))^2
        '''
        not_done = 1 - tf.cast(self.placeholders['done_mask'], tf.float32)
        q_target = self.placeholders['rewards'] + not_done * self.config.gamma * tf.reduce_max(self.target_q, axis=1)
        action_indices = tf.one_hot(self.placeholders['actions'], self.train_simulator.get_num_actions())
        q_est = tf.reduce_sum(self.q * action_indices, axis=1)
        self.loss = tf.reduce_mean((q_target - q_est) ** 2)


    def _add_update_target_op(self, q_scope, target_q_scope):
        '''
        Set self.update_target_op to copy the weights from the Q network to the
        target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes.

        Args
        - q_scope: str, name of the variable scope for Q network we are training
        - target_q_scope: str, name of the variable scope for the target Q network
        '''
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        assign_ops = [tf.assign(target_v, v) for target_v, v in zip(target_q_vars, q_vars)]
        self.update_target_op = tf.group(*assign_ops)
