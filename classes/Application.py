import numpy as np
import random

class Application:
    """
    Job: Associated with each user id and define
    - Job type, resource requirements, UE requirements, arrival, departure times
    """
    
    def __init__(self, job_type, user_id, time_steps, job_profiles):
        """
        job_type - integer [0,1,2] based on the sample profiles we have 
        user_id - associate job with user id
        """
        
        # Init values
        self.user_id = user_id
        self.job_type = job_type
        self.time_steps = time_steps
        self.job_profile = job_profiles[job_type] # Load rate, latency restriction      
       
        # Load latency/offload values
        self.latency_req = self.job_profile.latency_req
        self.offload_mean = self.job_profile.offload_mean
        
        # Record total amount of load generated per ts
        self.load_history = {}
        self.offload_history = {}
        self.queue_length = {} # key[server] [big_ts, small_ts, ts_taken]
        
        # Record Reinforcement learning values below (UCB, confidence range)
        
        
        # Keep information on where relevant VM is
        
    def new_load(self,ts_big,ts_small):
        """
        Return a load value for this timestep based on exponential distribution value
        This will be logged into the 
        """
        
        self.load_history[(ts_big,ts_small)] =  np.random.geometric(1/self.offload_mean)
        return
    
    def record_queue_length(self, queue_response, server, ts_big, ts_small):
        
        if server not in self.queue_length:
            self.queue_length[server] = np.empty([0,3])
            
        row = np.array([[ts_big, ts_small, queue_response]])
        self.queue_length[server] = np.append(self.queue_length[server], row, axis=0)
        
        return
        
    
    def offload_uniform(self, containers_deployed, ts_big, ts_small):
        
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
        
        self.offload_history[(ts_big,ts_small)] = int_load
        
        to_offload = {}
        
        for s in range(int_load.shape[0]):
            if int_load[s] > 0:
                to_offload[(s,self.job_type)] = np.array([[self.user_id,ts_small,int_load[s],int_load[s]]])
        
        return to_offload
    
    def offload_distance(self, containers_deployed, n):
        
        return

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