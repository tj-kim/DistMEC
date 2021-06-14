import numpy as np
import math
import copy

from classes.Application import *
from classes.User import *
from classes.Server import *
from classes.Central_Controller import *
from classes.Container import *

class Sim_Params:
    """
    Simulation params hold information about system setting for simulation
    - timestep - 5 min per timestep
    - length - 1 mile per unit length
    """
    
    def __init__(self, big_ts, small_ts, x_length, y_length, num_users, num_servers, num_apps, cc_mode = 'dist',
                 app_mode = 'dist'):
        
        self.big_ts = big_ts
        self.small_ts = small_ts
        self.x_length = x_length
        self.y_length = y_length
        self.num_users = num_users
        self.num_servers = num_servers
        self.num_apps = num_apps
        self.cc_mode = cc_mode
        self.app_mode = app_mode
        
        # Non-specified instances
        self.low_mean_jobs = 5
        self.high_mean_jobs = 15
        self.server_weak_range = np.array([[2,2]])
        self.server_strong_range = np.array([[2,2]])
        self.user_max_speed = 2.5
        self.user_lamdas = [1/0.7,1/2,1/0.7] # 3 mph, 10 mph, 20 mph
        self.user_num_path = 10
        self.container_service_low = 20
        self.container_service_high = 30
        self.deploy_rounds = 5
        self.dist_n = 1
        self.dist_p = 1
        
        
def setup_sim(sim_param):
    
    # Create Job Profiles
    num_app_types = sim_param.num_apps
    low_mean = sim_param.low_mean_jobs
    high_mean = sim_param.high_mean_jobs
    job_profiles = []

    for i in range(num_app_types):
        job_profiles += [Job_Profile(job_name = str(i),
                                     latency_req = 3,
                                     offload_mean = np.random.uniform(low_mean,high_mean))]


    # System physical Boundaries - All action takes within this
    boundaries = np.array([[0,sim_param.x_length],[0,sim_param.y_length]])


    # Generate Servers
    num_resource = 1
    weak_range = sim_param.server_weak_range
    strong_range = sim_param.server_strong_range

    # Generate Server
    servers = []
    idx_counter = 0

    for i in range(sim_param.num_servers):
        servers.append(Server(boundaries,level=2,rand_locs=True,locs=None))
        servers[-1].server_resources(num_resource, weak_range, strong_range)
        servers[-1].assign_id(idx_counter)
        idx_counter += 1


    # Generate Users
    users= []
    idx_counter = 0


    for i in range(sim_param.num_users):
        users += [User(boundaries, sim_param.big_ts, 2, sim_param.user_lamdas, sim_param.user_max_speed)]
        users[-1].generate_MC(servers)
        users[-1].assign_id(idx_counter)
        idx_counter += 1


    # Generate Apps
    num_apps = len(users)
    app_id = np.random.choice(num_app_types,num_apps)
    apps = []

    for i in range(len(app_id)):
        apps += [Application(job_type=app_id[i], user_id=i, 
                             time_steps=sim_param.big_ts, job_profiles=job_profiles, mode = sim_param.app_mode,
                             dist_n = sim_param.dist_n, dist_p = sim_param.dist_p, user = users[i],
                             ts_big = sim_param.big_ts, ts_small = sim_param.small_ts)]
        
    # Generate Containers - in dictionary indexed by {(server,app)}
    containers = {}
    
    for s in range(len(servers)):
        for a in range(num_app_types):
            service_rate = np.random.uniform(sim_param.container_service_low, sim_param.container_service_high)
            latency_restriction = job_profiles[a].latency_req
            containers[(s,a)] = Container(a, s, service_rate, latency_restriction)

    return servers, users, containers, apps