class QL_Config(object):
    train_num_steps = 200       # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 50             # how often to print loss and other training statistics (training steps)

    test_freq = 100             # how often to evaluate model (training steps)
    test_num_episodes = 30      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    # TODO: make these unique per experiment
    log_dir = 'logs/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'ckpts/'         # path to directory for checkpoints
    ckpt_freq = 10000

    gamma = 0.9                 # discount factor
    lr = 0.01                   # learning rate
    
    batch_size = 20             # batch size

    q_values_metrics_size = 1000
    target_update_freq = 1000
    replay_buffer_size = 100

    cheating = False
    helper_reward_factor = 0.
    helper_reward_hl = 40000

    eps_begin  = 1
    eps_end    = 0.1
    eps_delay = 20000
    eps_nsteps = train_num_steps/2
    test_epsilon = 0.0         # epsilon for test-time exploration (0 for always choosing the best action)

    num_players = 2
