import numpy as np
import copy

class Central_Controller:
    """
    Central controller solves P1 by placing VMs at desired servers
    Keeps track of latency based weights for users and inits them for VMs without history
    """
    
    def __init__(self, servers, containers, system_params, apps, users = None):
        
        """
        All inputs are lists of objects of their relevant type
        """
        
        self.servers = servers
        self.containers = containers
        self.apps = apps
        self.users = users
        self.mode = None
        
        # Extract information
        self.user_locations = None
        self.beta_weights = None
        
        # System settings
        self.ts_big = 0
        self.ts_small = 0
        
        # Extract from System Params
        self.num_app = system_params.num_apps
        self.num_user = system_params.num_users
        self.num_servers = len(self.servers)
        self.num_cores = self.obtain_num_cores()
        self.max_deployed = self.obtain_max_deployed()
        self.server_dists = self.server2server_dist(servers)
        self.dist_n = system_params.dist_n
        self.dist_p = system_params.dist_p
        
        # Initialize Utilities
        # self.container_utility = np.zeros(len(containers))
        # self.app_utility = np.zeros(self.num_app)
        self.container_deployed = self.VM_placement_init(self.apps) # 1- True, 0 - False
        self.beta_temp = np.zeros(self.container_deployed.shape)
        
    def obtain_num_cores(self):
        """
        Get the total number of VMs that can be deployed in the system
        on a per server basis
        """
        
        num_cores = np.ones(len(self.servers))
        
        for i in range(len(self.servers)):
            server = self.servers[i]
            num_cores[i] = int(server.avail_rsrc[0])
            
        return num_cores
    
    def obtain_max_deployed(self):
        """
        obtain the maximum allowed unique applications for each server?
        """
        
        num_cores = np.ones(len(self.servers))
        for i in range(len(self.servers)):
            server = self.servers[i]
            num_cores[i] = min(server.avail_rsrc[0],self.num_app)
            
        return num_cores
        
        
    def VM_placement(self, users, apps, rounds = 3):
        """
        solves P1 - The VM placement problem at every big TS
        inputs: users-apps are lists of nominal objects
                limit - number of iterations to run swapping before quitting
        """
        
        self.container_deployed = self.VM_placement_init(self.apps)
        utils, order, deployed_coor, array_utils = self.compute_container_utility(self.container_deployed,
                                                                                     users,apps, self.mode)
        
        for t in range(rounds):
            # Test the swapping
            utils,order, deployed_coor, array_utils = self.compute_container_utility(self.container_deployed,
                                                                                     users,apps, self.mode)
            deployed_coor_new = copy.deepcopy(deployed_coor)


            for i in order:
                curr_app = deployed_coor_new[i,0]
                curr_server = deployed_coor_new[i,1]

                # Check if there are other containers with that same in system
                compare = np.ones(deployed_coor_new[:,0].shape)*curr_app
                if np.argwhere(deployed_coor_new[:,0]==compare).flatten().shape[0] == 1:
                    continue

                # ELSE -  Perform swapping
                # Update new utilities based on container
                container_deployed_new = self.coor2array_containers(deployed_coor_new)
                _, _, _, curr_utils = self.compute_container_utility(container_deployed_new,users,apps,self.mode)

                # Obtain existing VM + app util for subject + app utils for all others
                vm_util_orig = curr_utils[deployed_coor_new[i,0],deployed_coor_new[i,1]]
                app_util_orig = np.sum(curr_utils[:,:],axis=1)

                # Make a list of all candidate VM's that are not already deployed at server
                already_deployed = deployed_coor_new[np.argwhere(deployed_coor_new[:,1]==curr_server).flatten()][:,0]
                # Make this parallel ***
                for replace_a in range(self.num_app):
                    if replace_a not in already_deployed:
                        # calculate vm_util and app util before and after 
                        prev_app = deployed_coor_new[i,0]
                        deployed_coor_temp = copy.deepcopy(deployed_coor_new)
                        deployed_coor_temp[i] = np.array([replace_a,curr_server])
                        container_deployed_temp = self.coor2array_containers(deployed_coor_temp)
                        _, _, _, curr_utils_temp = self.compute_container_utility(container_deployed_temp,users,apps,self.mode)
                        vm_util_temp = curr_utils_temp[deployed_coor_temp[i,0],deployed_coor_temp[i,1]]
                        app_util_temp = np.sum(curr_utils_temp[:,:],axis=1)

                        # If conditions are fulfilled, swap into deployed_coor_new
                        app_util_b4 = app_util_orig[prev_app]+app_util_orig[replace_a]
                        app_util_af = app_util_temp[prev_app]+app_util_temp[replace_a]
                        if (vm_util_temp > vm_util_orig) and (app_util_af >= app_util_b4):
                            deployed_coor_new = copy.deepcopy(deployed_coor_temp)
                            curr_utils = copy.deepcopy(curr_utils_temp)
                            vm_util_orig = copy.deepcopy(vm_util_temp)
                            app_util_orig = copy.deepcopy(app_util_orig)

            # print('util', np.sum(app_util_orig))    

            # Finalize util
            self.container_deployed = self.coor2array_containers(deployed_coor_new)
        
        return
        
    def VM_placement_init(self, apps):
        """
        Set aside proportional vm counts for app load
        Place them closest to most users in round robin style
        """
        
        # Get expected load of each application group
        load_product = np.zeros(self.num_app)
        
        for app in apps:
            job_type = app.job_type
            load_product[job_type] += app.offload_mean
                
        # Calculate number of VM for each app
        container_deployed = np.zeros([self.num_app, self.num_servers])
        space_available = copy.deepcopy(self.max_deployed)
        total_VM = np.sum(self.max_deployed)
        
        load_prop = load_product/np.sum(load_product)
        init_deployed = np.floor(load_prop * (total_VM - self.num_app)) + 1
        
        # Cut all init deployed below num_server
        edit_idx = np.argwhere(init_deployed>len(self.servers))
        init_deployed[edit_idx] = len(self.servers)
        
        # Deploy the servers from max-->min count
        deploy_order = np.argsort(load_prop)
        
        for app_idx in deploy_order:
            still_avail = np.subtract(space_available, np.sum(container_deployed,axis=0))
            avail_servers = np.argwhere(still_avail>0).flatten()
            num_VM = int(min(init_deployed[app_idx],avail_servers.shape[0]))
            
            deploy_servers = np.random.choice(avail_servers, size=num_VM, replace=False, p=None)
            container_deployed[app_idx,deploy_servers] = 1
            
        # Fill in the available cores with empty VMs
        still_avail = np.subtract(space_available, np.sum(container_deployed,axis=0))
        avail_s_idx = np.argwhere(still_avail>0).flatten()
                
        for s in avail_s_idx:
            deployed = container_deployed[:,s]
            candidates = np.argwhere(deployed == 0).flatten()            
            new_apps = np.random.choice(candidates,size= int(still_avail[s]),replace=False,p=None)
            
            container_deployed[new_apps,s] = 1
            
        
        return container_deployed
        
    def compute_container_utility(self, container_deployed, users, apps, mode = None):
        """
        Compute all utilities for containers that are deployed
        Sort the VMs by their utilities in ascending order
        """
        
        # Get all coordinates deployed
        deployed_coor = np.argwhere(container_deployed>0)
        utils = np.empty(0)
                
        offload_dict = self.offload_estimate(container_deployed,users,apps,self.ts_big, mode)    
        cost = -1 * self.latency_cost(offload_dict, users, self.ts_big, self.server_dists)

        for a in range(self.num_app):
            util_idx = np.argwhere(deployed_coor[:,0]==a).flatten()
            i0,i1 = deployed_coor[util_idx][:,0],deployed_coor[util_idx][:,1]
            util =  cost[i0,i1]
            utils = np.append(utils,util)
        
        return utils, np.argsort(utils), deployed_coor, cost

    
    def offload_estimate(self, container_deployed, users, apps, big_ts, mode = None):
        """
        Ratio of incoming traffic to be offloaded to different containers that are deployed
        """
        
        offload_dict = {} # Per user
        dist_offset = self.dist_n

        # Parallalize this process later:
        if mode == 'dist':
            for u in range(len(users)):
                app_id = apps[u].job_type
                user_offload_ratio = np.zeros(container_deployed.shape)
                app_row = container_deployed[app_id,:]
                
                # Dist^2 per 
                user_loc = int(users[u].user_voronoi[0,int(big_ts)])
                dists = self.server_dists[user_loc] + self.dist_n
                temp_row = (app_row*(dists**self.dist_p))
                
                for i in range(temp_row.shape[0]):
                    if temp_row[i]>0:
                        temp_row[i] = 1/temp_row[i]
                                
                app_row = temp_row/np.sum(temp_row)
                # print(app_row)
                user_offload_ratio[app_id,:] = app_row
                
                offload_dict[u] = user_offload_ratio
        else:
            for u in range(len(users)):
                app_id = apps[u].job_type
                user_offload_ratio = np.zeros(container_deployed.shape)
                app_row = container_deployed[app_id,:]
                app_row = app_row/np.sum(app_row)
                user_offload_ratio[app_id,:] = app_row

                offload_dict[u] = user_offload_ratio
        
        return offload_dict
    
    def latency_cost(self, offload_dict, users, big_ts, srv_dists):
        """
        calculate the 2nd half of the utility based on latency 
        """
        
        cost_as = np.zeros(offload_dict[0].shape)
        
        for u in range(len(users)):
            # 1. Obtain user location
            voronoi_idx = int(users[u].user_voronoi[0,int(big_ts)])
            dists = np.tile(srv_dists[voronoi_idx],(offload_dict[u].shape[0],1))
            u_cost = dists*offload_dict[u]
            cost_as += u_cost
            
        return cost_as
    
    def server2server_dist(self, servers):
        
        # make list of locations
        x,y = np.zeros(len(servers)),np.zeros(len(servers))
        for i in range(len(servers)):
            x[i], y[i] = servers[i].locs[0], servers[i].locs[1]
            
        # Compute euclidean distance for every combination of servers        
        srv_dists = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
        
        return srv_dists
    
    
    def coor2array_containers(self, deployed_coor_new):
        container_deployed_new = np.zeros(self.container_deployed.shape)
        x = deployed_coor_new[:,0]
        y = deployed_coor_new[:,1]
        container_deployed_new[x,y] = 1

        return container_deployed_new