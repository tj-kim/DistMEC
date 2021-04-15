import numpy as np
import copy

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
        self.num_cores = obtain_num_cores()
        self.max_deployed = obtain_max_deployed()
        
        # Initialize Utilities
        self.container_utility = np.zeros(len(containers))
        self.app_utility = np.zeros(self.num_app)
        self.container_deployed = np.zeros([num_app, len(servers)]) # 1- True, 0 - False
        
        
    def update_big_ts(self):
        
        self.ts_big += 1
        
    def update_small_ts(self):
        
        self.ts_small += 1
        
    def obtain_num_cores(self):
        """
        Get the total number of VMs that can be deployed in the system
        on a per server basis
        """
        
        num_cores = np.ones(len(servers))
        
        for i in range(len(servers)):
            server = servers[i]
            num_cores[i] = int(server.avail_rsrc[0])
            
        return num_cores
    
    def obtain_max_deployed(self):
        """
        obtain the maximum allowed unique applications for each server?
        """
        
        num_cores = np.ones(len(servers))
        for i in range(len(servers)):
            server = servers[i]
            num_cores[i] = min(server.avail_rsrc[0],self.num_app)
            
        return num_cores
        
        
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
        self.container_deployed = VM_placement_init(apps)
        
        # Calculate utility function for all placed containers and apps
        
        
        # Sort utility functions for both app/containers
        
        # Matching Iteratively - from lowest app --> container and swap
        
        return
        
    def VM_placement_init(self, apps):
        """
        Set aside proportional vm counts for app load
        Place them closest to most users in round robin style
        """
        
        # Get expected load of each application group
        load_product = np.zeros(self.num_app)
        
        for app in apps:
            job_type = app.job_types
            load_product[job_type] += app.offload_mean
        
        # Calculate number of VM for each app
        container_deployed = np.zeros([self.num_app, self.num_servers])
        space_available = copy.deepcopy(self.max_deployed)
        total_VM = np.sum(self.max_deployed)
        
        load_prop = load_product/np.sum(load_product)
        init_deployed = np.floor(load_prop * (total_VM- self.num_app)) + 1
        
        # Cut all init deployed below num_server
        edit_idx = np.argwhere(init_deployed>len(self.servers))
        init_deployed[edit_idx] = len(self.servers)
        
        # Deploy the servers from max-->min count
        deploy_order = np.argsort(load_prop)
        
        for idx in deploy_order:
            avail_servers = np.argwhere(space_available>0)
            num_VM = min(init_deployed[idx],avail_servers.shape[0])
            
            deploy_servers = np.random.choice(avail_servers, size=num_VM, replace=False, p=None)
            container_deployed[idx,:] = np.reshape(deploy_servers,container_deployed[idx,:].shape)
            
        # Fill in the available cores with empty VMs
        still_avail = np.subtract(space_available, np.sum(container_deployed,axis=0))
        avail_s_idx = np.argwhere(still_avail>0)
        
        for s in avail_s_idx:
            deployed = container_deployed[:,s]
            candidates = np.argwhere(deployed == 0)
            
            new_apps = np.random.choice(candidates,size=still_avail[s],replace=False,p=None)
            
            container_deployed[new_apps,s] = 1
        
            
        
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
    
