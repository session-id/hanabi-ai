from models.base_model import QL_Model
import tensorflow as tf


class LinearQL_Model(QL_Model):

    def _get_q_values_op(self, scope):
        '''
        Args
        - scope: str, name of scope
        '''
        # TODO: this is for you, Arthur!
        raise NotImplementedError