from models.base_model import QL_Model
import tensorflow as tf


class DQL_Model(QL_Model):

    def _get_q_values_op(self, scope):
        '''
        Args
        - state: tf.Tensor, shape [batch_size, state_dim]
        - scope: str, name of scope

        Returns
        - q: tf.Tensor, shape [batch_size, num_actions]
        '''
        # TODO: this is for you, Arthur!
        raise NotImplementedError