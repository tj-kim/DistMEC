import numpy as np
import copy

class Container:
    """
    Container: Associated with a specific server-application pair for one-to-many matching
    Here 
    - Job type, resource requirements, UE requirements, arrival, departure times
    """
    
    def __init__(self, app_id, server_id, service_rate, latency_restriction):
        """
        app_id - which application this container is for
        server_id - which server this VM is designated for (binary on off)
        """
        
        self.app_id = app_id
        self.server_id = server_id
        self.deployed = False # True when active at server
        self.service_rate = service_rate
        self.latency_restriction = latency_restriction
        
        # queue --> [user_id, ts_arrive, load, remaining load]
        self.queue = np.empty((0,4))
        # history --> [user_id,ts_arrive,load,completion_time]
        self.history = np.empty((0,4))
        
    def add_to_queue(self, new_offload):
        """
        At the start of small TS add all queues
        new_offload -> np array of shape (1,3)
        """
        
        num_jobs = new_offload.shape[0]
        
        # Add new arrival information
        self.queue = np.append(self.queue,new_offload,axis=0)
        # Compute run time for each job
        loads = self.queue[:,3]
        load_cm = np.cumsum(loads)
        service_time = (load_cm/self.service_rate)[-num_jobs:]
        new_loads = loads[-num_jobs:]
        
        # Add to history
        new_history = copy.deepcopy(new_offload)
        new_history[:,3] = service_time
        
        self.history = np.append(self.history, new_history,axis=0)
        
        st_temp = np.append(new_history[:,0].reshape(new_history[:,0].shape[0],1),
                                     new_loads.reshape(new_loads.shape[0],1), axis=1)
                                             
                                     
        service_time_log= np.append(st_temp, service_time.reshape(service_time.shape[0],1),axis=1)
        
        return service_time_log
    
    def serve_ts(self):
        """
        subtract from queue based on the existing service rate
        Update self.queue and whatever remains
        """
        
        remaining_service = copy.deepcopy(self.service_rate)
        while remaining_service > 0:
            if self.queue.shape[0] > 0:
                remainder = self.queue[0,3]
                if remainder <= remaining_service:
                    self.queue = np.delete(self.queue,0,0)
                elif remainder > remaining_service:
                    self.queue[0,3] = remainder - remaining_service

                remaining_service -= remainder
            else: 
                remaining_service = 0
        
        return
        
    def calc_emp_beta(self):
        """
        Calculate the emprical value of beta based on latency violations
        """
        
        return