class QL_Config(object):
    train_num_steps = 200       # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 5              # how often to print loss and other training statistics (training steps)

    test_freq = 100             # how often to evaluate model (training steps)
    test_num_episodes = 30      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    # TODO: make these unique per experiment
    log_dir = 'logs/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'ckpts/'         # path to directory for checkpoints

    gamma = 0.9                 # discount factor
    lr = 0.01                   # learning rate
    soft_epsilon = 0.01         # epsilon for e-greedy exploration
    
    batch_size = 20             # batch size

    q_values_metrics_size = 1000
    replay_buffer_size = 100


class LinearQL_Config(object):
    train_num_steps = 20000     # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 5              # how often to print loss and other training statistics (training steps)

    test_freq = 100             # how often to evaluate model (training steps)
    test_num_episodes = 30      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    # TODO: make these unique per experiment
    log_dir = 'logs/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'ckpts/'         # path to directory for checkpoints

    gamma = 1                   # discount factor
    lr = 0.01                   # learning rate
    soft_epsilon = 0.01         # epsilon for e-greedy exploration
    
    batch_size = 20             # batch size

    widths = [30, 30]           # widths if you want to do mlp stuff

    q_values_metrics_size = 1000
    target_update_freq = 100
    replay_buffer_size = 100


class DeepQL_Config(object):
    train_num_steps = 50000       # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 200              # how often to print loss and other training statistics (training steps)

    test_freq = 1000             # how often to evaluate model (training steps)
    test_num_episodes = 30      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    log_dir = 'deep-logs/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'deep-ckpts/'         # path to directory for checkpoints

    gamma = 0.99                 # discount factor
    lr = 0.01                   # learning rate
    soft_epsilon = 0.01         # epsilon for e-greedy exploration
    
    batch_size = 20             # batch size

    widths = [10,10]           # widths if you want to do mlp stuff

    q_values_metrics_size = 1000
    target_update_freq = 10
    replay_buffer_size = 100
