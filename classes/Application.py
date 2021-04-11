import numpy as np

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
        self.load_history = np.zeros(time_steps)
        
        # Record Reinforcement learning values below (UCB, confidence range)
        
    def new_load(self,t):
        """
        Return a load value for this timestep based on exponential distribution value
        This will be logged into the 
        """
        
        self.load_history[t] =  np.random.exponential(1/self.offload_mean)
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