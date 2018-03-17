import tensorflow as tf

class config():
  

    # output config
    output_path  = "results/" + "pg/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    nsteps_train       = 20
    eps_begin          = 1.0
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    
    # model and training config
    num_batches = 100 # number of batches trained on 
    batch_size = 1000 # number of steps used to compute each policy update
    max_ep_len = 1000 # maximum episode length
    learning_rate = 3e-2
    gamma              = .99 # the discount factor
    use_baseline = True
    normalize_advantage=True 
    # parameters for the policy and baseline models
    n_layers = 1
    layer_size = 16
    activation=tf.nn.relu 


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
