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
        
        self.user_id = user_id
        self.job_type = job_type
        self.time_steps = time_steps
        self.job_profile = job_profiles[job_type]      
       
        # Extract Values from user
        self.mvmt_class = None
        self.refresh_rate = None


class Job_Profile:
    """
    Make list of job profiles with
    - UE properties (how much to offload generated per small ts)
    - Length Properties
    - Resource Consumption
    """
    
    def __init__(self, job_name,
                    latency_req_range,
                    offload_range):
        """
        Add job profile to list 
        """
        
        self.job_name = job_name
        self.latency_req_range = latency_req_range # milisecond
        self.offload_range = offload_range