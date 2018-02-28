from models.base_model import QL_Model
import tensorflow as tf

class DQL_Model(QL_Model):

    def __init__(self, inputs, config, train_simulator, test_simulator):
        super(QL_Model, self).__init__(
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

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        # build the graph
        self.build()


    def get_action(self, state):
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values


    def save(self):
        ckpt_prefix = os.path.join(self.config.ckpt_dir, 'ckpt')
        self.saver.save(self.sess, ckpt_prefix)


    def train_step(self, step, replay_buffer, return_):
        batch = replay_buffer.sample(self.config.batch_size)



    def build(self):
        # self.placeholders: dict, {str => tf.placeholder}
        self.add_placeholders()

        # self.loss
        self._add_loss_op()

        # self.update_target_op
        self._add_update_target_op()

        # self.train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])
        self.train_op = optimizer.minimize(self.loss)


    def _add_summaries(self, scope):
        '''
        Args
        - scope: str, name of scope
        '''
        with tf.variable_scope(scope):
            summary_names = ['avg_reward', 'max_reward', 'std_reward', 'avg_q', 'max_q', 'std_q']
            self.summary_placeholders = {}
            for name in summary_names:
                self.summary_placeholders[name] = tf.placeholder(tf.float32, shape=[], name=name)
                tf.summary.scalar(name, self.summary_placeholders[name])
            tf.summary.scalar("loss", self.loss)
            self.summaries[scope]


    def _add_placeholders(self):
        self.placeholders = {
            'states'      = tf.placeholder(tf.float32, [None, self.train_simulator.state_dim]),
            'actions'     = tf.placeholder(tf.int32,   [None]),
            'rewards'     = tf.placeholder(tf.float32, [None]),
            'states_next' = tf.placeholder(tf.float32, [None, self.train_simulator.state_dim]),
            'done_mask'   = tf.placeholder(tf.bool,    [None]),
            'lr'          = tf.placeholder(tf.float32, [])
        }


    def _get_q_values_op(self, scope):
        '''
        Args
        - scope: str, name of scope
        '''
        pass


    def _add_loss_op(self):
        '''
        Sets self.loss to the loss operation defined as

        Q_samp(s) = r if done
                  = r + gamma * max_a' Q_target(s', a')
        loss = (Q_samp(s) - Q(s, a))^2 
        '''
        not_done = 1 - tf.cast(self.placeholders['done_mask'], tf.float32)
        q_target = self.placeholders['rewards'] + not_done * self.config.gamma * tf.reduce_max(self.target_q, axis=1)
        action_indices = tf.one_hot(self.placeholders['actions'], self.train_simulator.num_actions)
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
