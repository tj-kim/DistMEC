import numpy as np
import math
import copy

from classes.Application import *
from classes.User import *
from classes.Server import *
from solver.Sim_Params import *
from classes.Central_Controller import *
from classes.Container import *

def run_sim_temp(sim_param, servers, users, containers, apps):

    # Loop through big time step
    cc = Central_Controller(servers, containers, sim_param, apps, users)
    cc.mode = sim_param.cc_mode
    cc_deployment_history = {}

    for bt in range(sim_param.big_ts):

        # Deploy the containers to the servers
        cc.big_ts = bt
        cc.VM_placement(users,apps,sim_param.deploy_rounds)
        cc_deployment_history[bt] = cc.container_deployed

        # For each small time step offload and serve at container
        for st in range(sim_param.small_ts):
            # random order between users when offloading for each app
            cc.small_ts = st
            usr_order = np.arange(len(users))
            np.random.shuffle(usr_order)

            temp_containers = {}
            queue_replies = {}

            # Make offloading decision
            for u in usr_order:
                # Generate load
                # apps[u].new_load(ts_big=bt,ts_small=st)
                # Decide to offload given servers --> add offload policy to app class
                offload_u = apps[u].offload(cc.container_deployed, bt, st, cc)
                for (s,a) in offload_u.keys():
                    if (s,a) not in temp_containers:
                        temp_containers[(s,a)] = np.empty([0,4])
                    temp_containers[(s,a)] = np.append(temp_containers[(s,a)],offload_u[(s,a)],axis=0)

            # Scramble arrived job and add to queue, apps record latency
            for (s,a) in temp_containers.keys():
                sa_offload = temp_containers[(s,a)]
                np.random.shuffle(sa_offload) 
                replies = containers[(s,a)].add_to_queue(sa_offload)
                # print(containers[(s,a)].queue)
                queue_replies[(s,a)] = replies

                for i in range(replies.shape[0]):
                    # Add distance between app and server
                    a_id, reply_len, load = int(replies[i,0]), replies[i,2], replies[i,1]
                    dist = cc.server_dists[int(users[a_id].user_voronoi_true[bt]),s]
                    apps[a_id].record_queue_length(reply_len, s, bt, st, load, dist)

                # Service the queue
                containers[(s,a)].serve_ts()

    return apps