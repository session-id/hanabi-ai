import json
import os
import time
from collections import deque
import numpy as np
import tensorflow as tf

from models.replay_buffer import ReplayBuffer
from models.base_model import QL_Model


class Recurrent_QL_Model(QL_Model):
    '''
    Abstract Q-learning model with support for dynamically sized recurrent input.

    Subclasses must implement the _get_q_values_op() method.
    '''
        
    def get_action(self, state, dynamic_state, valid_actions_mask, epsilon=0):
        '''
        Performs epsilon-greedy action selection.

        Args
        - state: np.array, shape [state_dim]
        - dynamic_state: np.array, shape [seq_len, dynamic_state_dim]
        - valid_actions_mask: np.array, shape [num_actions]
        - epsilon: float in [0, 1], probability of selecting a random action
            - set to 0 to always choose the best action

        Returns: (action, q_values)
        - action: int, index of the action to take
        - q_values: np.array, type float32, vector of Q-values for the given state
        '''
        q_values = self.sess.run(self.q, feed_dict={
            self.placeholders['states']: [state],
            self.placeholders['dynamic_state']: [dynamic_state],
            self.placeholders['valid_actions_mask']: [valid_actions_mask]
        })[0]
        if np.random.random() < epsilon:
            valid_actions = np.where(valid_actions_mask == True)[0]
            random_action = np.random.choice(valid_actions)
            return random_action, q_values
        else:
            return np.argmax(q_values), q_values

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

                feature_dict = self.train_simulator.get_state_vector(state)
                states = feature_dict['states']
                dynamic_states = feature_dict['dynamic_states']
                valid_actions_mask = np.zeros(num_actions, dtype=bool)
                valid_action_indices = list(self.train_simulator.get_valid_actions(state))
                valid_actions_mask[valid_action_indices] = True

                action, q_values = self.get_action(states, dynamic_states, valid_actions_mask, epsilon=epsilon)
                decay = 0.5 ** (float(step) / self.config.helper_reward_hl)
                new_state, reward, done = self.train_simulator.take_action(state, action,
                        bomb_reward=self.config.bomb_reward * decay,
                        alive_reward = self.config.alive_reward * decay)

                valid_actions_next_mask = np.zeros(num_actions, dtype=bool)
                valid_action_indices = list(self.train_simulator.get_valid_actions(new_state))
                valid_actions_next_mask[valid_action_indices] = True

                new_feature_dict = self.train_simulator.get_state_vector(new_state, cheat=self.config.cheating)
                replay_buffer.store(step, states, valid_actions_mask, action, reward, done, new_features, valid_actions_next_mask)

                state = new_state
                total_reward += reward
                self.update_averages('train', ep_reward=None, q_values=q_values)

                if step > self.config.train_start:
                    if step % self.config.test_freq == 0:
                        test_rewards = self.evaluate(step=step)
                        test_avg_rewards.append(np.mean(test_rewards))

                    if step % self.config.print_freq == 0:
                        loss = self.train_step(step, replay_buffer, return_stats=True)
                        duration = time.time() - start_time
                        print('step {:6d}, episode {:4d}, epsilon: {:0.4f}, loss: {:0.4f}, time: {:0.3f}s'.format(
                            step, episode, epsilon, loss, duration))
                    else:
                        self.train_step(step, replay_buffer, return_stats=False)

                step += 1
                if done or step >= self.config.train_num_steps:
                    break

            episode += 1
            self.update_averages('train', ep_reward=total_reward, q_values=None)

            # save checkpoints (if ckpt_freq is defined)
            if self.config.ckpt_freq and episode % self.config.ckpt_freq == 0:
                self.save()
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
                    print(self.test_simulator.get_action_names(state)[action])
                if step is not None:
                    self.update_averages('test', ep_reward=None, q_values=q_values)
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards[ep] = total_reward

            if step is not None:
                self.update_averages('test', ep_reward=total_reward, q_values=None)

        if self.config.num_test_to_print > 0:
            print("")

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, std_reward / np.sqrt(len(rewards)))
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
            self.placeholders['dynamic_state']:      batch['dynamic_state'],
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
                self.summary_placeholders['q_values']: self.metrics_train['q_values']
            }

            # copy the summary feed_dict into feed_dict
            for k, v in summary_fd.items():
                feed_dict[k] = v

            _, loss, train_summary_str  = self.sess.run([self.train_op, self.loss, self.summaries_train],
                feed_dict=feed_dict)
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

        # compute Q values
        self.q = self._get_q_values_wrapper(self.placeholders['states'], self.placeholders['dyanmic_states'], scope="q")
        self.target_q = self._get_q_values_wrapper(self.placeholders['states_next'], self.placeholders['dyanmic_states'], scope="target_q")

        # self.update_target_op
        self._add_update_target_op("q", "target_q")

        # self.loss
        self._add_loss_op()

        # self.train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])
        self.train_op = optimizer.minimize(self.loss)

        # self.summary_placeholders, self.summaries_train, self.summaries_test
        self._add_summaries()


    def _add_placeholders(self):
        state_dim = self.train_simulator.get_state_vector_size()
        num_actions = self.train_simulator.get_num_actions()
        self.placeholders = {
            'states'      : tf.placeholder(tf.float32, shape=[None, state_dim]),
            'dynamic_states'      : tf.placeholder(tf.float32, shape=[None, max_seq_len, state_dim]),
            'actions'     : tf.placeholder(tf.int32,   shape=[None]),
            'rewards'     : tf.placeholder(tf.float32, shape=[None]),
            'states_next' : tf.placeholder(tf.float32, shape=[None, state_dim]),
            'lr'          : tf.placeholder(tf.float32, shape=[]),
            'done_mask'   : tf.placeholder(tf.bool,    shape=[None]),

            # boolean mask of valid actions given the current batch of states
            'valid_actions_mask' : tf.placeholder(tf.bool, shape=[None, num_actions]),
            'valid_actions_next_mask' : tf.placeholder(tf.bool, shape=[None, num_actions])
        }
