import json
import os
import time
from collections import deque
import numpy as np
import tensorflow as tf

from models.replay_buffer import ReplayBuffer


class RL_Model(object):
    def __init__(self, config, train_simulator, test_simulator):
        self.config = config
        self.train_simulator = train_simulator
        self.test_simulator = test_simulator

    def get_action(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self, step):
        raise NotImplementedError


class QL_Model(RL_Model):
    '''
    Abstract Q-learning model.

    Subclasses must implement the _get_q_values_op() method.
    '''

    def __init__(self, config, train_simulator, test_simulator, eps_decay, ckpt_prefix=None):
        '''
        Args
        - config
        - train_simulator
        - test_simulator
        - eps_decay: function that takes a time step and returns an epsilon value
        '''
        super(QL_Model, self).__init__(
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
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.sess = tf.Session()

        # build the graph
        self.build()

        # initialize variables
        if ckpt_prefix is not None:
            self.load(ckpt_prefix)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # Synchronize networks
            self.sess.run(self.update_target_op)

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)

        # set the epsilon
        self.eps_decay = eps_decay

        with open(os.path.join(config.log_dir, 'config.txt'), 'w') as f:
            json.dump({ x: getattr(config,x) for x in dir(config) if not x.startswith("__")}, f)
        with open(os.path.join(config.log_dir, 'read_config.txt'), 'w') as f:
            f.write(sorted([ (x, getattr(config,x)) for x in dir(config) if not x.startswith("__")]).__repr__())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total parameters:", total_parameters)

        self.best_avg_reward = 0.0
        self.best_avg_reward2 = 0.0
        self.best_std2 = 0.0


    def _get_q_values_wrapper(self, state, valid_actions_mask, scope):
        '''
        Args
        - state: tf.Tensor, shape [batch_size, state_dim]
        - scope: str, name of scope, one of ['q', 'target_q']

        Returns
        - q: tf.Tensor, shape [batch_size, num_actions], with any invalid actions having a large negative q-value
        '''
        assert scope in ['q', 'target_q']
        q_values = self._get_q_values_op(state, scope)
        neg_inf = tf.fill(tf.shape(q_values), -10.0**10)
        q = tf.where(valid_actions_mask, x=q_values, y=neg_inf)
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


    def save(self, step):
        '''
        Saves the weights of the current model to config.ckpt_dir with prefix 'ckpt'
        '''
        ckpt_prefix = os.path.join(self.config.ckpt_dir, 'ckpt')
        self.saver.save(self.sess, ckpt_prefix, global_step=step)

        
    def load(self, fileprefix):
        '''
        Loads model from saved state. Fileprefix should not include the ckpt_dir.
        '''
        self.saver = tf.train.import_meta_graph(os.path.join(self.config.ckpt_dir, fileprefix)+'.meta')
        self.saver.restore(self.sess, os.path.join(self.config.ckpt_dir, fileprefix))
            
        
    def get_action(self, state, valid_actions_mask, epsilon=0):
        '''
        Performs epsilon-greedy action selection.

        Args
        - state: np.array, shape [state_dim]
        - valid_actions_mask: np.array, shape [num_actions]
        - epsilon: float in [0, 1], probability of selecting a random action
            - set to 0 to always choose the best action

        Returns: (action, q_values)
        - action: int, index of the action to take
        - q_values: np.array, type float32, vector of Q-values for the given state
        '''
        q_values = self.sess.run(self.q, feed_dict={
            self.placeholders['states']: [state],
            self.placeholders['valid_actions_mask']: [valid_actions_mask]
        })[0]
        if np.random.random() < epsilon:
            valid_actions = np.where(valid_actions_mask == True)[0]
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
            metrics_dict['rewards'].append(ep_reward)
        if q_values is not None:
            metrics_dict['q_values'].extend(q_values)


    def train(self):
        '''Train the model

        Returns
        - test_avg_rewards: list of float, avg rewards from self.evaluate() every config.test_freq steps
        '''
        test_avg_rewards = []

        # initialize replay buffer
        replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
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
                epsilon = self.eps_decay(step)

                features = self.train_simulator.get_state_vector(state, cheat=self.config.cheating)
                valid_actions_mask = np.zeros(num_actions, dtype=bool)
                valid_action_indices = list(self.train_simulator.get_valid_actions(state))
                valid_actions_mask[valid_action_indices] = True

                action, q_values = self.get_action(features, valid_actions_mask, epsilon=epsilon)
                valid_q_values = q_values[valid_action_indices]
                decay = 0.5 ** (float(step) / self.config.helper_reward_hl)
                new_state, reward, done = self.train_simulator.take_action(state, action,
                        bomb_reward=self.config.bomb_reward * decay,
                        alive_reward = self.config.alive_reward * decay)

                valid_actions_next_mask = np.zeros(num_actions, dtype=bool)
                valid_action_indices = list(self.train_simulator.get_valid_actions(new_state))
                valid_actions_next_mask[valid_action_indices] = True

                new_features = self.train_simulator.get_state_vector(new_state, cheat=self.config.cheating)
                replay_buffer.store(step, features, valid_actions_mask, action, reward, done, new_features, valid_actions_next_mask)

                state = new_state
                total_reward += reward

                self.update_averages('train', ep_reward=None, q_values=valid_q_values)

                if step > self.config.train_start:
                    if step % self.config.test_freq == 0:
                        test_rewards = self.evaluate(step=step)
                        test_avg_rewards.append(np.mean(test_rewards))

                    if step % self.config.print_freq == 0:
                        loss = self.train_step(step, replay_buffer, epsilon, return_stats=True)
                        duration = time.time() - start_time
                        print('step {:6d}, episode {:4d}, epsilon: {:0.4f}, loss: {:0.4f}, time: {:0.3f}s'.format(
                            step, episode, epsilon, loss, duration))
                    else:
                        self.train_step(step, replay_buffer, epsilon, return_stats=False)

                step += 1
                if done or step >= self.config.train_num_steps:
                    break

            episode += 1
            self.update_averages('train', ep_reward=total_reward, q_values=None)

            # save checkpoints (if ckpt_freq is defined)
            if self.config.ckpt_freq and episode % self.config.ckpt_freq == 0:
                self.save(step)
                # a little hacky, but save episode number in a txt file in the same directory
                # (for later: set episode number correctly when loading from checkpoint)
                with open(os.path.join(self.config.ckpt_dir,'ckpt_epsisode.txt'), 'w') as f:
                    f.write('last saved epsiode: {}'.format(episode))

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
        num_actions = self.test_simulator.get_num_actions()
        rewards = np.zeros(num_episodes, dtype=np.float64)

        def eval_rewards(update=True):
            for ep in range(self.config.test_num_episodes):
                if ep < self.config.num_test_to_print:
                    print("\nTest episode: {}".format(ep))
                total_reward = 0
                state = self.test_simulator.get_start_state()

                while True:
                    if ep < self.config.num_test_to_print:
                        state.print_self()
                    features = self.train_simulator.get_state_vector(state, cheat=self.config.cheating)
                    valid_actions_mask = np.zeros(num_actions, dtype=bool)
                    valid_action_indices = list(self.test_simulator.get_valid_actions(state))
                    valid_actions_mask[valid_action_indices] = True

                    action, q_values = self.get_action(features, valid_actions_mask, epsilon=self.config.test_epsilon)
                    state, reward, done = self.test_simulator.take_action(state, action)
                    if ep < self.config.num_test_to_print:
                        print('\t' + self.test_simulator.get_action_names(state)[action])
                    if step is not None and update:
                        valid_q_values = q_values[valid_action_indices]
                        self.update_averages('test', ep_reward=None, q_values=valid_q_values)
                    total_reward += reward
                    if done:
                        break

                # updates to perform at the end of an episode
                rewards[ep] = total_reward

                if step is not None and update:
                    self.update_averages('test', ep_reward=total_reward, q_values=None)

            if self.config.num_test_to_print > 0:
                print("")

            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards) / np.sqrt(len(rewards))  # estimate of population std-dev. = sample std-dev / sample_size

            return avg_reward, std_reward

        avg_reward, std_reward = eval_rewards(update=True)

        if avg_reward > self.best_avg_reward:
            print("New best average reward!")
            self.save(step)
            self.best_avg_reward = avg_reward
            self.best_avg_reward2, self.best_std2 = eval_rewards(update=False)

        msg = "Average reward: {:04.2f} +/- {:04.2f}. Best reward: {:04.2f} +/- {:04.2f}.".format(
            avg_reward, std_reward, self.best_avg_reward2, self.best_std2)
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


    def train_step(self, step, replay_buffer, epsilon, return_stats=False):
        '''
        Run a training step

        Args
        - step: int
        - replay_buffer: ReplayBuffer
        - epsilon: float, e-greedy parameter between 0 and 1
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
            self.placeholders['valid_actions_next_mask']: batch['valid_actions_next_mask'],
            self.placeholders['done_mask']:          batch['done_mask'],
            self.placeholders['lr']: self.config.lr
        }

        if return_stats:
            # write the summary to TensorBoard
            summary_fd = {
                self.summary_placeholders['rewards']: self.metrics_train['rewards'],
                self.summary_placeholders['q_values']: self.metrics_train['q_values'],
                self.summary_placeholders['epsilon']: epsilon
            }

            # copy the summary feed_dict into feed_dict
            for k, v in summary_fd.items():
                feed_dict[k] = v

            _, loss, train_summary_str  = self.sess.run([self.train_op, self.losses['total'], self.summaries_train],
                feed_dict=feed_dict)
            self.summary_writer.add_summary(train_summary_str, step)
            self.summary_writer.flush()
            return loss

        else:
            self.sess.run(self.train_op, feed_dict=feed_dict)

        # occasionaly update target network with q network
        if step % self.config.target_update_freq == 0:
            self.sess.run(self.update_target_op)


    def _add_optimizer_op(self, scope):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grad_var_pairs = optimizer.compute_gradients(self.losses['total'], var_list=var_list)

        if self.config.grad_clip:
            grad_var_pairs = [
                (tf.clip_by_norm(grad, clip_norm=self.config.clip_val), v) for grad, v in grad_var_pairs
            ]
            self.train_op = optimizer.apply_gradients(grad_var_pairs)
        else:
            self.train_op = optimizer.minimize(self.losses['total'])

        self.weights_norm = tf.global_norm([v for v in var_list])
        self.grad_norm = tf.global_norm([grad for grad, _ in grad_var_pairs])


    def build(self):
        # self.placeholders: dict, {str => tf.placeholder}
        self._add_placeholders()

        # compute Q values
        self.q = self._get_q_values_wrapper(self.placeholders['states'],
                self.placeholders['valid_actions_mask'], scope="q")
        self.double_q = self._get_q_values_wrapper(self.placeholders['states_next'],
                self.placeholders['valid_actions_next_mask'], scope="q")
        self.target_q = self._get_q_values_wrapper(self.placeholders['states_next'],
                self.placeholders['valid_actions_next_mask'], scope="target_q")

        # self.update_target_op
        self._add_update_target_op(q_scope="q", target_q_scope="target_q")

        # self.losses
        self._add_loss_op()

        # self.train_op, self.grad_norm, self.weights_norm
        self._add_optimizer_op(scope="q")

        # self.summary_placeholders, self.summaries_train, self.summaries_test
        self._add_summaries()


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
            'rewards': deque(maxlen=self.config.test_num_episodes),
            'q_values': deque(maxlen=self.config.q_values_metrics_size)
        }
        self.metrics_test = {
            'rewards': deque(maxlen=self.config.test_num_episodes),
            'q_values': deque(maxlen=self.config.q_values_metrics_size)
        }
        self.summary_placeholders = {
            'rewards': tf.placeholder(tf.float32, shape=[None], name='rewards'),
            'q_values': tf.placeholder(tf.float32, shape=[None], name='q_values'),
            'epsilon': tf.placeholder(tf.float32, shape=[], name='epsilon')
        }

        avg_reward, var_reward = tf.nn.moments(self.summary_placeholders['rewards'], axes=[0])
        avg_q, var_q = tf.nn.moments(self.summary_placeholders['q_values'], axes=[0])

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
            ] + [
                tf.summary.scalar("loss_total", self.losses['total']),
                tf.summary.scalar("loss_mse", self.losses['mse']),
                tf.summary.scalar("loss_reg", self.losses['reg']),
                tf.summary.scalar("grad_norm", self.grad_norm),
                tf.summary.scalar("weights_norm", self.weights_norm),
                tf.summary.scalar("epsilon", self.summary_placeholders['epsilon'])
            ])

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
            'valid_actions_mask' : tf.placeholder(tf.bool, shape=[None, num_actions]),
            'valid_actions_next_mask' : tf.placeholder(tf.bool, shape=[None, num_actions])
        }


    def _add_loss_op(self):
        '''
        Sets self.losses to the loss operation defined as
            loss = (y - Q(s, a))^2

        # normal DQN
        y = r if done
          = r + gamma * max_a' Q_target(s', a')

        # double DQN
        y = r if done
          = r + gamma * Q_target(s', argmax_a' Q(s', a'))

        Note: self.losses requires the 'done_mask', 'rewards', 'actions', and 'states_next'
            placeholders to be filled
        '''
        num_actions = self.train_simulator.get_num_actions()
        not_done = 1 - tf.cast(self.placeholders['done_mask'], tf.float32)

        if self.config.use_double_q:
            best_next_actions = tf.one_hot(tf.argmax(self.double_q, axis=1), num_actions)
        else:
            best_next_actions = tf.one_hot(tf.argmax(self.target_q, axis=1), num_actions)
        y = self.placeholders['rewards'] + not_done * self.config.gamma * \
            tf.reduce_sum(self.target_q * best_next_actions, axis=1)

        action_indices = tf.one_hot(self.placeholders['actions'], num_actions)
        q_est = tf.reduce_sum(self.q * action_indices, axis=1)
        loss_mse = tf.losses.mean_squared_error(y, q_est, reduction=tf.losses.Reduction.MEAN)

        loss_total = tf.losses.get_total_loss(add_regularization_losses=True)
        loss_reg = tf.losses.get_regularization_loss()

        self.losses = {
            'mse': loss_mse,
            'total': loss_total,
            'reg': loss_reg
        }


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
