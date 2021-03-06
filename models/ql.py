from models.base_model import QL_Model
import tensorflow as tf


class MLP_QL_Model(QL_Model):

    def _get_q_values_op(self, states, scope):
        '''
        Args
        - state: tf.Tensor, shape [batch_size, state_dim]
        - scope: str, name of scope

        Returns
        - q: tf.Tensor, shape [batch_size, num_actions]
        '''
        h = states
        num_actions = self.train_simulator.get_num_actions()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for width in self.config.widths:
                h = tf.layers.dense(h, width,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.fc_reg)
                )
            q_values = tf.layers.dense(h, num_actions, activation=None)
        return q_values
