class PredictDSRNNConfigs:
    predict_steps = 5
    pred_timestep =  0.25 ## matching for flatland step_size
    
    time_step = 0.25 ## matching for pedsim hz
    pred_interval = int(pred_timestep // time_step) 
    load_path = "prediction_data/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj"    
    device = "cuda"