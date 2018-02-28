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

    def train(self):
        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer() # TODO

        episode = 0
        step = 0

        # train for self.config.train_num_steps steps
        while step < self.config.train_num_steps:
            total_reward = 0
            state = self.train_simulator.reset()

            # train multiple episodes
            while True:
                start_time = time.time()

                action = self.get_action(state)
                new_state, reward, done = self.train_simulator.take_action(state, action)
                replay_buffer.store(state, action, reward, done)
                state = new_state
                total_reward += reward

                if step > self.config.train_start:
                    if step % self.config.eval_freq == 0:
                        self.evaluate()

                    if step % self.config.print_freq == 0:
                        # TODO: print Max Q, Max Reward, Avg Reward, Epsilon (for e-greedy), learning rate, gradient norm
                        loss = self.train_step(step, replay_buffer, return_stats=True)
                        duration = time.time() - start_time
                        print('Step {:05d}. Episode {:02d}. loss: {:0.4f}, time: {:0.3f}s'.format(
                            step, epsiode, loss, duration))
                    else:
                        self.train_step(step, replay_buffer, return_stats=False)

                step += 1
                if done or step >= self.config.train_num_steps:
                    break

            episode += 1
            episode_rewards.append(total_reward)

        # evaluate again at the end of training
        self.evaluate()


    def evaluate(self):
        '''Evaluate model

        Returns
        - avg_reward: float, average reward on self.config.test_num_episodes
        '''
        num_episodes = self.config.test_num_episodes
        rewards = np.zeros(num_episodes, dtype=np.float64)

        for ep in range(self.config.test_num_episodes):
            total_reward = 0
            state = self.test_simulator.reset()

            while True:
                action = self.get_action(state)
                state, reward, done = self.test_simulator.take_action(state, action)
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards[ep] = total_reward

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, std_reward)
        print(msg)
        return avg_reward


    def train_step(self, step, replay_buffer, return_stats=False):
        '''
        Perform training step

        Args
        - step: int, step of training
        - replay_buffer: ReplayBuffer, buffer for sampling
        - return_stats: bool, whether or not to calculate and return training statistics (e.g. loss)

        Returns
        - if return_stats=False, then None
        - otherwise:
          - loss: float
        '''
        raise NotImplementedError
