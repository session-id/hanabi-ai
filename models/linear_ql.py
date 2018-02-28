from models.base_model import QL_Model
import tensorflow as tf


class LinearQL_Model(QL_Model):

    def _get_q_values_op(self, scope):
        '''
        Args
        - scope: str, name of scope
        '''
        # TODO: this is for you, Arthur!
        h = self.placeholders['states']
        num_actions = self.train_simulator.num_actions()
        with tf.variable_scope(scope):
            for width in widths:
                h = tf.layers.dense(h, width, activation=tf.nn.relu())
            q_values = tf.layers.dense(h, num_actions)
        return q_values
