import numpy as np

class Central_Controller:
    """
    Central controller solves P1 by placing VMs at desired servers
    Keeps track of latency based weights for users and inits them for VMs without history
    """
    
    def __init__(self, servers, containers, system_params):
        
        """
        All inputs are lists of objects of their relevant type
        """
        
        self.servers = servers
        self.containers = containers
        
        # Extract information
        self.user_locations = None
        self.beta_weights = None
        
        # System settings
        self.ts_big = 0
        self.ts_small = 0
        
        # Extract from System Params
        self.num_app = system_params.num_app
        self.num_user = system_param.num_user
        
        # Initialize Utilities
        self.container_utility = np.zeros(len(containers))
        self.app_utility = np.zeros(self.num_app)
        self.container_deployed = np.zeros([len(servers),num_app]) # 1- True, 0 - False
        
        
    def update_big_ts(self):
        
        self.ts_big += 1
        
    def update_small_ts(self):
        
        self.ts_small += 1
        
    def VM_placement(self, users, apps, limit = None):
        """
        solves P1 - The VM placement problem at every big TS
        inputs: users-apps are lists of nominal objects
                limit - number of iterations to run swapping before quitting
        """
        
        # Flush utility functions
        self.container_utility = np.zeros(len(containers))
        self.app_utility = np.zeros(self.num_app)
        self.container_deployed = np.zeros([len(servers),num_app]) 
        
        # Initialize the Container placements on VMs 
        self.container_deployed, self.containers = VM_placement_init
        
        # Calculate utility function for all placed containers and apps
        
        
        # Sort utility functions for both app/containers
        
        # Matching Iteratively - from lowest app --> container and swap
        
        return
        
    def VM_placement_init(self, users, apps):
        """
        Set aside proportional vm counts for app load
        Place them closest to most users in round robin style
        """
        
        # Across all aplications take expected load and bin them (at least 1 each)
        
        # Round robin style place VM randomly
        
        return container_deployed, containers
        
    def compute_container_utility(self, users, apps, container):
        """
        Compute all utilities for containers that are deployed
        """
        
        return
        
    def compute_app_utility(self, users, apps, containers):
        """
        Compute utilities for all apps given 
        AKA sum all the utilities for all apps 
        """
        
        return
    
