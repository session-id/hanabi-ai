class LinearQL_Config(object):
    run_id = '16'

    train_num_steps = 200000     # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 50             # how often to print loss and other training statistics (training steps)

    test_freq = 1000             # how often to evaluate model (training steps)
    test_num_episodes = 30      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    # TODO: make these unique per experiment
    log_dir = 'logs/' + run_id +'/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'ckpts/' + run_id +'/'         # path to directory for checkpoints

    gamma = 0.99                # discount factor
    lr = 0.00003                   # learning rate
    
    batch_size = 100             # batch size

    widths = [50, 50, 50]           # widths if you want to do mlp stuff

    q_values_metrics_size = 1000
    target_update_freq = 1000
    replay_buffer_size = 1000

    ckpt_freq = 10000

    eps_begin  = 1
    eps_end    = 0.1
    eps_delay = 20000
    eps_nsteps = train_num_steps/2
    test_epsilon = 0.0         # epsilon for test-time exploration (0 for always choosing the best action)

    num_test_to_print = 1       # number of test trajectories to print. set to 0 to never print

    cheating = False
    helper_reward_factor = 2.
    helper_reward_hl = 2000000000

    num_players = 2


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
    
    batch_size = 20             # batch size

    widths = []           # widths if you want to do mlp stuff

    q_values_metrics_size = 1000
    target_update_freq = 1000
    replay_buffer_size = 100

    eps_begin  = 1
    eps_end    = 0.01
    eps_nsteps = train_num_steps/2
    test_epsilon = 0.01         # epsilon for test-time exploration (0 for always choosing the best action)

    num_test_to_print = 1       # number of test trajectories to print. set to 0 to never print
