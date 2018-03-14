class Config(object):
    run_id = 'easier_widths_target_10000_bomb_2.0_alive_0.1_lr_1e-4_eps_1.0_0.2_80000_double_q_1_replay_100_hl_50000_run_1'

    train_num_steps = 400000     # number of steps for training
    train_start = 0             # how many steps of observation before beginning training
    print_freq = 50             # how often to print loss and other training statistics (training steps)

    test_freq = 1000             # how often to evaluate model (training steps)
    test_num_episodes = 200      # how many episodes to test for

    gpu = -1                    # which GPU to use. set to -1 to use CPU only

    # TODO: make these unique per experiment
    log_dir = 'logs/' + run_id +'/'           # path to directory for TensorBoard summaries
    ckpt_dir = 'ckpts/' + run_id +'/'         # path to directory for checkpoints

    gamma = 0.99                # discount factor
    lr = 1e-4                   # learning rate
    grad_clip = False
    
    batch_size = 100             # batch size

    widths = []           # widths if you want to do mlp stuff

    q_values_metrics_size = 1000
    target_update_freq = 10000
    replay_buffer_size = 100
    use_double_q = True

    ckpt_freq = 10000

    eps_begin  = 1.0
    eps_end    = 0.2
    eps_nsteps = 80000 # train_num_steps/2
    test_epsilon = 0.0         # epsilon for test-time exploration (0 for always choosing the best action)

    num_test_to_print = 1       # number of test trajectories to print. set to 0 to never print

    cheating = False
    bomb_reward = -2.0
    alive_reward = 0.1
    helper_reward_hl = 50000

    num_players = 2
    colors = ['red', 'white', 'blue']
    cards_per_player = 3
    max_number = 3
    number_counts = [3, 2, 2]
