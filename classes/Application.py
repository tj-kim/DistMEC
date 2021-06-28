import numpy as np
import random

class Application:
    """
    Job: Associated with each user id and define
    - Job type, resource requirements, UE requirements, arrival, departure times
    """
    
    def __init__(self, job_type, user_id, time_steps, job_profiles, mode, dist_n,dist_p, user,
                ts_big, ts_small):
        """
        job_type - integer [0,1,2] based on the sample profiles we have 
        user_id - associate job with user id
        """
        
        # Init values
        self.user_id = user_id
        self.job_type = job_type
        self.time_steps = time_steps
        self.job_profile = job_profiles[job_type] # Load rate, latency restriction
        self.user = user
        self.latency_threshold = self.job_profile.latency_req
       
        # Load latency/offload values
        self.latency_req = self.job_profile.latency_req
        self.offload_mean = self.job_profile.offload_mean
        self.mode = mode # dist vs. uniform
        self.dist_n = dist_n
        self.dist_p = dist_p
        self.offload_mode = 'd' # 'i'
        
        # Record total amount of load generated per ts
        self.load_history = {}
        self.queue_length = {} # key[server] [big_ts, small_ts, load, ts_taken, distance, max(ts_taken-thresh,0)]
        
        # Record Reinforcement learning values below (UCB, confidence range)
        
        
        # Keep information on where relevant VM is

        # Generate load ahead of time
        self.new_load(ts_big,ts_small)
        
    def new_load(self,ts_big,ts_small):
        """
        Return a load value for this timestep based on exponential distribution value
        This will be logged into the 
        """
        
        for tb in range(ts_big):
            for ts in range(ts_small):
                self.load_history[(tb,ts)] =  np.random.geometric(1/self.offload_mean)
        return
    
    def record_queue_length(self, queue_response, server, ts_big, ts_small, load, s_dist):
        
        num_params = 6
        
        if server not in self.queue_length:
            self.queue_length[server] = np.empty([0,num_params])
            
        row = np.array([[ts_big, ts_small, load, queue_response, s_dist, np.maximum(queue_response-self.latency_threshold,0)]])
        self.queue_length[server] = np.append(self.queue_length[server], row, axis=0)
        
        return
        
    
    def offload_uniform(self, containers_deployed, ts_big, ts_small):
        # Get rid of this (not needed as distance case can cover this fully)
        
        valid_containers = containers_deployed[self.job_type]
        num_deployed = np.sum(valid_containers)
        load = self.load_history[(ts_big,ts_small)]
        
        double_load = valid_containers/num_deployed * load
        int_load = np.floor(double_load)
        diff = load - np.sum(int_load)
        deployed = np.where(valid_containers==1)[0]
        deploy_choice = random.choices(list(deployed),k=int(diff))
        
        for i in deploy_choice:
            int_load[i] += 1
                
        to_offload = {}
        
        for s in range(int_load.shape[0]):
            if int_load[s] > 0:
                to_offload[(s,self.job_type)] = np.array([[self.user_id,ts_small,int_load[s],int_load[s]]])
        
        return to_offload
    
    def offload_distance(self, containers_deployed, ts_big, ts_small, user, n,p, central_controller):
        """
        n- distance offset value
        p- distance power value
        """
        
        valid_containers = containers_deployed[self.job_type]
        num_deployed = np.sum(valid_containers)
        load = self.load_history[(ts_big,ts_small)]
        
        # distribute based on distance then run same int code
        user_loc = int(user.user_voronoi_true[int(ts_big)])
        dists = central_controller.server_dists[user_loc] + n
        temp_row = (valid_containers*(dists**p))

        for i in range(temp_row.shape[0]):
            if temp_row[i]>0:
                temp_row[i] = 1/temp_row[i]

        app_row = temp_row/np.sum(temp_row)

        double_load = app_row * load
        to_offload = {}
        
        if self.offload_mode == 'i':
            int_load = np.floor(double_load)
            diff = load - np.sum(int_load)
            deployed = np.where(valid_containers==1)[0]
            deploy_choice = random.choices(list(deployed),k=int(diff))

            for i in deploy_choice:
                int_load[i] += 1

            for s in range(int_load.shape[0]):
                if int_load[s] > 0:
                    to_offload[(s,self.job_type)] = np.array([[self.user_id,ts_small,int_load[s],int_load[s]]])
        elif self.offload_mode == 'd':
            
            # Optional Print for Debugging
            # print(double_load)
            
            for s in range(double_load.shape[0]):
                if double_load[s] > 0:
                    to_offload[(s,self.job_type)] = np.array([[self.user_id,ts_small,double_load[s],double_load[s]]])
        
        return to_offload
    
    def offload(self, containers_deployed, ts_big, ts_small, central_controller):
        
        if self.mode =='dist':
            to_offload = self.offload_distance(containers_deployed, ts_big, ts_small, 
                                               self.user, self.dist_n, self.dist_p, central_controller)
        else:
            to_offload = self.offload_uniform(containers_deployed, ts_big, ts_small)
            
        return to_offload
        

    def cmab_round(self, arm_idx, arm_info, t):
        """
        Run CMAB to select how much load to offload to each arm
        """
        
        return
        

class Job_Profile:
    """
    Make list of job profiles with
    - UE properties (how much to offload generated per small ts)
    - Length Properties
    - Resource Consumption
    """
    
    def __init__(self, job_name,
                    latency_req,
                    offload_mean):
        """
        Add job profile to list 
        """
        
        self.job_name = job_name
        self.latency_req = latency_req # milisecond
        self.offload_mean = offload_mean