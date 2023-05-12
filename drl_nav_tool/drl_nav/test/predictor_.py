import sys, os
from os.path import join, isdir
import pickle
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/train/gst_updated/')
from src.gumbel_social_transformer.st_model import st_model

def seq_to_graph(seq_, seq_rel, attn_mech='rel_conv'):
    """
    inputs:
        - seq_ # (n_env, num_peds, 2, obs_seq_len)
        - seq_rel # (n_env, num_peds, 2, obs_seq_len)
    outputs:
        - V # (n_env, obs_seq_len, num_peds, 2)
        - A # (n_env, obs_seq_len, num_peds, num_peds, 2)
    """
    V = seq_rel.permute(0, 3, 1, 2) # (n_env, obs_seq_len, num_peds, 2)
    seq_permute = seq_.permute(0, 3, 1, 2) # (n_env, obs_seq_len, num_peds, 2)
    A = seq_permute.unsqueeze(3)-seq_permute.unsqueeze(2) # (n_env, obs_seq_len, num_peds, 1, 2) - (n_env, obs_seq_len, 1, num_peds, 2)
    return V, A

#CrowdNavPredInterfaceMultiEnv
class Predictor():
    def __init__(self, load_path, device, config):
    # *** Load model
        self.args = config
        self.device = device
        # self.nenv = num_env

        # Uncomment if you want a fixed random seed.
        # torch.manual_seed(args_eval.random_seed)
        # np.random.seed(args_eval.random_seed)
        self.args_eval = config
        checkpoint_dir = join(load_path, 'checkpoint')
        self.model = st_model(self.args_eval, device=device).to(device)
        model_filename = 'epoch_'+str(self.args_eval.num_epochs)+'.pt'
        model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()

    def forward(self, input_traj,input_binary_mask, sampling = True):
        """
        inputs:
            - input_traj:
                # numpy
                # (n_env, num_peds, obs_seq_len, 2)
            - input_binary_mask:
                # numpy
                # (n_env, num_peds, obs_seq_len, 1)
                # Zhe: I think we should not just have the binary mask of shape (n_env, number of pedestrains, 1)
                # because some agents are partially detected, and they should not be simply ignored.
            - sampling:
                # bool
                # True means you sample from Gaussian.
                # False means you choose to use the mean of Gaussian as output.
        outputs:
            - output_traj:
                # torch "cpu"
                # (n_env, num_peds, pred_seq_len, 5)
                # where 5 includes [mu_x, mu_y, sigma_x, sigma_y, correlation coefficient]
            - output_binary_mask:
                # torch "cpu"
                # (n_env, num_peds, 1)
                # Zhe: this means for prediction, if an agent does not show up in the last and second
                # last observation time step, then the agent will not be predicted.
        """

        invalid_value = -999.
        # *** Process input data
        obs_traj = input_traj.permute(0,1,3,2) # (n_env, num_peds, 2, obs_seq_len)
        n_env, num_peds = obs_traj.shape[:2]
        loss_mask_obs = input_binary_mask[:,:,:,0] # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_obs = loss_mask_obs[:,:,:-1] * loss_mask_obs[:,:,-1:]
        loss_mask_rel_obs = torch.cat((loss_mask_obs[:,:,:1], loss_mask_rel_obs), dim=2) # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_pred = (torch.ones((n_env, num_peds, self.args_eval.pred_seq_len), device=self.device) * loss_mask_rel_obs[:,:,-1:])
        loss_mask_rel = torch.cat((loss_mask_rel_obs, loss_mask_rel_pred), dim=2) # (n_env, num_peds, seq_len)
        loss_mask_pred = loss_mask_rel_pred
        loss_mask_rel_obs_permute = loss_mask_rel_obs.permute(0,2,1).reshape(n_env*self.args_eval.obs_seq_len, num_peds) # (n_env*obs_seq_len, num_peds)
        attn_mask_obs = torch.bmm(loss_mask_rel_obs_permute.unsqueeze(2), loss_mask_rel_obs_permute.unsqueeze(1)) # (n_env*obs_seq_len, num_peds, num_peds)
        attn_mask_obs = attn_mask_obs.reshape(n_env, self.args_eval.obs_seq_len, num_peds, num_peds)
        
        obs_traj_rel = obs_traj[:,:,:,1:] - obs_traj[:,:,:,:-1]
        obs_traj_rel = torch.cat((torch.zeros(n_env, num_peds, 2, 1, device=self.device), obs_traj_rel), dim=3)
        obs_traj_rel = invalid_value*torch.ones_like(obs_traj_rel)*(1-loss_mask_rel_obs.unsqueeze(2)) \
            + obs_traj_rel*loss_mask_rel_obs.unsqueeze(2)
        v_obs, A_obs = seq_to_graph(obs_traj, obs_traj_rel, attn_mech='rel_conv')
        # *** Perform trajectory prediction
        sampling = False
        with torch.no_grad():
            v_obs, A_obs, attn_mask_obs, loss_mask_rel = \
                v_obs.to(self.device), A_obs.to(self.device), attn_mask_obs.to(self.device), loss_mask_rel.to(self.device)
            results = self.model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=0.03, hard=True, sampling=sampling, device=self.device)
            gaussian_params_pred, x_sample_pred, info = results
        mu, sx, sy, corr = gaussian_params_pred
        mu = mu.cumsum(1)
        sx_squared = sx**2.
        sy_squared = sy**2.
        corr_sx_sy = corr*sx*sy
        sx_squared_cumsum = sx_squared.cumsum(1)
        sy_squared_cumsum = sy_squared.cumsum(1)
        corr_sx_sy_cumsum = corr_sx_sy.cumsum(1)
        sx_cumsum = sx_squared_cumsum**(1./2)
        sy_cumsum = sy_squared_cumsum**(1./2)
        corr_cumsum = corr_sx_sy_cumsum/(sx_cumsum*sy_cumsum)
        mu_cumsum = mu.detach().to(self.device) + obs_traj.permute(0,3,1,2)[:,-1:]# np.transpose(obs_traj[:,:,:,-1:], (0,3,1,2)) # (batch, time, node, 2)
        mu_cumsum = mu_cumsum * loss_mask_pred.permute(0,2,1).unsqueeze(-1) + invalid_value*(1-loss_mask_pred.permute(0,2,1).unsqueeze(-1))
        output_traj = torch.cat((mu_cumsum.detach().to(self.device), sx_cumsum.detach().to(self.device), sy_cumsum.detach().to(self.device), corr_cumsum.detach().to(self.device)), dim=3)
        output_traj = output_traj.permute(0, 2, 1, 3) # (n_env, num_peds, pred_seq_len, 5)
        output_binary_mask = loss_mask_pred[:,:,:1].detach().to(self.device) # (n_env, num_peds, 1) # first step same as following in prediction
        return output_traj, output_binary_mask
    
#VecPretextNormalize
class Predict_Trajectory():
    def __init__(self, config = None):
        self.config = config
        load_path = os.path.join(os.getcwd(), self.config.pred.model_dir)
        if not os.path.isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)
        self.device=torch.device("cpu")
        self.config = config
        self.predictor = Predictor(load_path=load_path, device=self.device, config = self.args)
        self.num_envs = 1
        self.max_human_num = config.sim.human_num 
        self.pred_interval = int(self.config.data.pred_timestep//self.config.env.time_step)
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1

        # self.traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*999),
        #                          maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        # self.mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
        #                          maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)
    def reset(self,obs):
        # queue for inputs to the pred model
        # fill the queue with dummy values
        self.traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*999),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)

        self.step_counter = 0

        # for calculating the displacement of human positions
        self.last_pos = torch.zeros(self.num_envs, self.max_human_num, 2).to(self.device)
        obs, _ = self.process_obs_rew(obs, np.zeros(self.num_envs))

        return obs
    
    def process_obs_rew(self, O, rews=0.):
        # O: robot_node: [nenv, 1, 7], spatial_edges: [nenv, observed_human_num, 2*(1+predict_steps)],temporal_edges: [nenv, 1, 2],
        # pos_mask: [nenv, max_human_num], pos_disp_mask: [nenv, max_human_num]
        # prepare inputs for pred_model
        # find humans' absolute positions
        human_pos = O['robot_node'][:, :, :2] + O['spatial_edges'][:, :, :2]
        # insert the new ob to deque
        self.traj_buffer.append(human_pos)
        self.mask_buffer.append(O['visible_masks'].unsqueeze(-1))
        
        # [obs_seq_len, nenv, max_human_num, 2] -> [nenv, max_human_num, obs_seq_len, 2]
        # for i in range(len(self.mask_buffer)):
        #     print(len(self.mask_buffer[i]))

        in_traj = torch.stack(list(self.traj_buffer)).permute(1, 2, 0, 3)
        in_mask = torch.stack(list(self.mask_buffer)).permute(1, 2, 0, 3).float()

        # index select the input traj and input mask for GST
        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval]

        # forward predictor model
        out_traj, out_mask = self.predictor.forward(input_traj=in_traj, input_binary_mask=in_mask)
        out_mask = out_mask.bool()

        # add penalties if the robot collides with predicted future pos of humans
        # deterministic reward, only uses mu_x, mu_y and a predefined radius
        # constant radius of each personal zone circle
        # [nenv, human_num, predict_steps]
        hr_dist_future = out_traj[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
        # [nenv, human_num, predict_steps]
        collision_idx = torch.norm(hr_dist_future, dim=-1) < self.config.robot.radius + self.config.humans.radius

        # [1,1, predict_steps]
        # mask out invalid predictions
        # [nenv, human_num, predict_steps] AND [nenv, human_num, 1]
        collision_idx = torch.logical_and(collision_idx, out_mask)
        coefficients = 2. ** torch.arange(2, self.config.sim.predict_steps + 2, device=self.device).reshape(
            (1, 1, self.config.sim.predict_steps))  # 4, 8, 16, 32, 64

        # [1, 1, predict_steps]
        collision_penalties = self.config.reward.collision_penalty / coefficients

        # [nenv, human_num, predict_steps]
        reward_future = collision_idx.to(torch.float)*collision_penalties
        # [nenv, human_num, predict_steps] -> [nenv, human_num*predict_steps] -> [nenv,]
        # keep the values & discard indices
        reward_future, _ = torch.min(reward_future.reshape(self.num_envs, -1), dim=1)
        # print(reward_future)
        # seems that rews is on cpu
        rews = rews + reward_future.reshape(self.num_envs, 1).cpu().numpy()

        # get observation back to env
        robot_pos = O['robot_node'][:, :, :2].unsqueeze(1)

        # convert from positions in world frame to robot frame
        out_traj[:, :, :, :2] = out_traj[:, :, :, :2] - robot_pos

        # only take mu_x and mu_y
        out_mask = out_mask.repeat(1, 1, self.config.sim.predict_steps * 2)
        new_spatial_edges = out_traj[:, :, :, :2].reshape(self.num_envs, self.max_human_num, -1)
        O['spatial_edges'][:, :, 2:][out_mask] = new_spatial_edges[out_mask]

        # sort all humans by distance to robot
        # [nenv, human_num]
        hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
        sorted_idx = torch.argsort(hr_dist_cur, dim=1)
        # sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, 2*(self.config.sim.predict_steps+1))
        for i in range(self.num_envs):
            O['spatial_edges'][i] = O['spatial_edges'][i][sorted_idx[i]]

        obs={'robot_node':O['robot_node'],
            'spatial_edges':O['spatial_edges'],
            'temporal_edges':O['temporal_edges'],
            'visible_masks':O['visible_masks'],
             'detected_human_num': O['detected_human_num'],

        }

        self.last_pos = copy.deepcopy(human_pos)
        self.step_counter = self.step_counter + 1

        return obs, rews
