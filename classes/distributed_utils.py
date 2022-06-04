"""
Utils.py

Update Date: 6/2/22

Summary:
helper functions to generate simulation testbed, and run rounds of reservation based distributed UCB
"""

import numpy as np
import itertools
from classes.solver import *
# from classes.Server import *

def gen_eq_locs(space_1d, nums, offset = 0.5):
    # Generate well spread out locations in square space
    num_across = int(np.floor(np.sqrt(nums)))
    locs = []
    
    inc = space_1d/nums
    
    for i,j in itertools.product(range(num_across), range(num_across)):
        locs += [(i*inc+offset, j*inc+offset)]
    
    return locs

def obtain_w(Users, num_users, num_svrs): # checked
    
    w_curr = np.zeros([num_users,num_svrs])
    for i in range(num_users):
        w_curr[i] = Users[i].reward_scale[Users[i].usr_place]
    
    return w_curr

def update_user_locs(Users): # Checked
    
    for i in range(len(Users)):
        Users[i].next_loc()
    return
    
def get_arms_list(Users): # Checked
    arms = []
    for i in range(len(Users)):
        arms+= [Users[i].choose_arm()]
    return arms

def sort_server_results(arms_list, Servers, Users):

    reserve_id_dict = {}
    reserve_max_val_dict = {}
    reserve_time_dict = {}
    reward_dict = {}
    collision_flag_dict = {}

    for s in range(len(Servers)):
        usr_idxs = np.argwhere(np.array(arms_list) == s).flatten()
        scales = np.zeros(usr_idxs.shape[0])
        w_est = np.zeros(usr_idxs.shape[0])
        stay_times = np.zeros(usr_idxs.shape[0])
        for u in range(usr_idxs.shape[0]):
            scales[u] = Users[usr_idxs[u]].reward_scale[Users[usr_idxs[u]].usr_place,s]
            w_est[u] =  Users[usr_idxs[u]].ucb_raw[s]
            stay_times[u] = Users[usr_idxs[u]].expected_time

        user_list = usr_idxs.tolist()
        scales_list = scales.tolist()
        w_est_list = w_est.tolist()
        stay_times_list = stay_times.tolist()

        s_result = Servers[s].receive_users(user_list, scales_list, w_est_list, stay_times_list, len(Users))
        reserve_id, reserve_max_val, reserve_time, reward, collision_flag = s_result[0],s_result[1],s_result[2],s_result[3],s_result[4]
        reserve_id_dict[s] = reserve_id
        reserve_max_val_dict[s] = reserve_max_val
        reserve_time_dict[s] = reserve_time
        reward_dict[s] = reward
        collision_flag_dict[s] = collision_flag
    
    return reserve_id_dict,reserve_max_val_dict ,reserve_time_dict ,reward_dict , collision_flag_dict


def update_user_info(Users, arms_list, reserve_id_dict,reserve_max_val_dict ,
                     reserve_time_dict ,reward_dict ,collision_flag_dict, reservation_mode = True):
    # update UCB information from user 
    for u in range(len(Users)):
        arm_id = arms_list[u]
        reward = reward_dict[arm_id]
        collision_flag = collision_flag_dict[arm_id]
        max_reward = reserve_max_val_dict[arm_id]
        wait_time = reserve_time_dict[arm_id]
        chosen_idx = reserve_id_dict[arm_id]
        Users[u].receive_reward(arm_id, reward, collision_flag, max_reward, wait_time, 
                                chosen_idx, reservation_mode)
    return


def expected_reward_collision_sensing(arms, mus, w):
    exp_mus = np.zeros(len(arms))
    collision_counter = 0
    seen = []
    for i in range(len(arms)):
        num_simul_pulls = np.argwhere(np.array(arms)==arms[i]).flatten().shape[0]
        if num_simul_pulls == 1:
            exp_mus[i] = w[i, arms[i]]* mus[i, arms[i]]
        else:
            collision_counter += 1
        
    return np.sum(exp_mus), collision_counter

def get_user_locs(Users):
    usr_loc_list = []
    for i in range(len(Users)):
        usr_loc_list += [Users[i].usr_place]
        
    return usr_loc_list

def explore_rounds(Users, num_users, Servers, mu, regret, collision_count, 
                   optimal_reward = None, usr_move_flag = False, rounds=1):

    arms = list(range(num_users)) 
    num_svrs = len(Servers)
    
    for j in range(rounds):
        for i in range(num_svrs):
            w = obtain_w(Users, num_users, num_svrs)
            optimal = offline_optimal_action(w, mu)
            
            if optimal_reward is not None:
                optimal_reward[j*(num_svrs) + i] = optimal[1]
            
            reward_exp_now, collision_count[j*(num_svrs) + i] = expected_reward_collision_sensing(arms, mu, w)
            regret[j*(num_svrs) + i] = optimal[1] - reward_exp_now

            svr_res = sort_server_results(arms, Servers, Users)
            update_user_info(Users, arms, svr_res[0], svr_res[1], svr_res[2], svr_res[3], svr_res[4])
            if usr_move_flag:
                update_user_locs(Users)

            arms = sweep_init_next(arms, num_svrs)
    
    return
    
def play_round(Users, Servers, mu, regret, collision_count, 
               usr_move_flag = False, reservation_mode = True, debugger = False, optimal = None):
    
    num_users = len(Users)
    num_svrs = len(Servers)
    t = int(np.sum(Users[0].pulls))
    
    w = obtain_w(Users, num_users, num_svrs)
    
    if optimal == None:
        optimal = offline_optimal_action(w, mu)
    
    
    if debugger:
        print("time:", t)
        
        print("\nmu")
        print(mu)
    
        print("\nest mu")
        for i in range(len(Users)):
            print(Users[i].param_summed/Users[i].pulls)

        print("\nuser w")
        for i in range(len(Users)):
            print(Users[i].reward_scale[Users[i].usr_place])

        print("\nscaled_reward")    
        for i in range(len(Users)):
            print(Users[i].reward_scale[Users[i].usr_place] * mu[i])
            
        print("\nraw ucb")
        for i in range(len(Users)):
            print(Users[i].ucb_raw)
            
        print("\nscaled_ucb")    
        for i in range(len(Users)):
            print(Users[i].reward_scale[Users[i].usr_place] * Users[i].ucb_raw)

        print("\nuser pulls")
        for i in range(len(Users)):
            print(Users[i].pulls)

        print("\nuser locs")
        locci = []
        for i in range(len(Users)):
            locci += [Users[i].usr_place]
        print(locci)

    
    arms = get_arms_list(Users)
    reward_exp_now, collision_count[t] = expected_reward_collision_sensing(arms, mu, w)
    regret[t] = optimal[1] - reward_exp_now
    svr_res = sort_server_results(arms, Servers, Users)
    update_user_info(Users, arms, svr_res[0], svr_res[1], svr_res[2], svr_res[3], svr_res[4],
                     reservation_mode)
    if usr_move_flag:
        update_user_locs(Users)
        
    if debugger:
        
        print("\noptimal arms")
        print(optimal[0])
        
        print("\nchosen arms")
        print(arms)
        
        # Advanced
        print("\nmax log")    
        for i in range(len(Users)):
            print(Users[i].max_logs)
            
        print("\nwait times")    
        for i in range(len(Users)):
            print(Users[i].wait_times)
        
        print("\nregret")
        print(regret[t])
            
#     return
    return svr_res