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
        
        self.queue = 
        
        
    def add_to_queue: