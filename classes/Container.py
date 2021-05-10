import numpy as np

class Container:
    """
    Container: Associated with a specific server-application pair for one-to-many matching
    Here 
    - Job type, resource requirements, UE requirements, arrival, departure times
    """
    
    def __init__(self, app_id, server_id, service_rate):
        """
        app_id - which application this container is for
        server_id - which server this VM is designated for (binary on off)
        """
        
        self.app_id = app_id
        self.server_id = server_id
        self.deployed = False # True when active at server
        self.service_rate = service_rate
        
        # queue --> [user_id, job_id, load, remaining load]
        self.queue = np.empty((0,4))
        # history --> [user_id,job_id,load,completion_time,latency_restrict]
        self.history = np.empty((0,5))
        
        
        
    def add_to_queue(self, new_offload):
        """
        At the start of small TS add all queues
        new_offload -> np array of shape (1,3)
        """
        
        new_offload = np.reshape(new_offload, (1,3))
        
        return
        
    def calc_emp_beta(self):
        """
        Calculate the emprical value of beta based on latency violations
        """