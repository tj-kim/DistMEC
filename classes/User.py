import numpy as np
import itertools
import random

class User():

    def __init__(self, locs, svr_locs, max_dist = 7, threshold_dist = 6, self_weight = 0.5, P = None, ceiling = 20):
        
        self.locs = locs
        self.dists = self.get_dists()
        self.svr_locs = svr_locs
        self.ceiling = ceiling
        
        if P is None:
            self.P = self.make_P(threshold_dist, self_weight)
        else:
            self.P = P
            
        self.reward_dists = self.get_reward_dists()
        self.reward_scale = self.get_scales(max_dist)
        self.usr_place = self.init_loc()
        self.expected_time = self.get_expected_time()
    
    def make_P(self, threshold_dist, self_weight):
        # Creating Markov Transition Probability Matrix 
        
        P = np.zeros(self.dists.shape)
        locs = self.locs
        for i in range(len(locs)):
            cut_list = self.dists[i,:]
            others = np.squeeze(np.argwhere((cut_list > 0) * (cut_list < threshold_dist) == True))
            num_others = others.shape[0]
        
            # Draw values to make up row of MC
            self_transition = np.random.exponential(scale=1/self_weight)
            others_transition = np.random.exponential(scale=1/((1-self_weight)*num_others),size=num_others)
            total = self_transition + np.sum(others_transition)
            
            P[i,i] = self_transition/total
            
            idx = 0
            for j in others:
                P[i,j] = others_transition[idx]/total
                idx += 1
            
        return P
    
    def get_dists(self):
        # Obtaining distance matrix (from loc to loc) 
        
        locs = self.locs
        
        num_locs = len(locs)
        dists = np.zeros([num_locs,num_locs])
        
        for i,j in itertools.product(range(num_locs), range(num_locs)):
            if dists[i,j] == 0 and i != j:
                a = np.array(locs[i])
                b = np.array(locs[j])
                dists[i,j] = np.linalg.norm(a-b)
                dists[j,i] = dists[i,j]
        
        return dists
    
    def get_reward_dists(self):
        
        locs = self.locs
        svr_locs = self.svr_locs
        
        dists = np.zeros([len(locs),len(svr_locs)])
        
        for i,j in itertools.product(range(len(locs)), range(len(svr_locs))):
            a = np.array(locs[i])
            b = np.array(svr_locs[j])
            dists[i,j] = np.linalg.norm(a-b)
        
        return dists
    
    def get_scales(self,max_dist):
        # Mapping reward to [0,1] based on distance and max acceptable distance
        
        reward_scale = np.ones(self.reward_dists.shape) - self.reward_dists/max_dist
        reward_scale[reward_scale < 0] = 0
        
        return reward_scale
    
    def init_loc(self):
        # Initial location user takes 
        curr_loc = np.random.randint(0, len(self.locs)-1)
        return curr_loc
    
    def next_loc(self):
        # Update user location based on markov chain
        weights = self.P[self.usr_place]
        population = range(weights.shape[0])
        self.usr_place =  random.choices(population, weights)[0]
        self.expected_time = self.get_expected_time()
        
    def get_expected_time(self):
        # Get number of expected ts user will stay at this location
        try:
            curr_prob = np.ceil( 1/(1 - self.P[self.usr_place, self.usr_place]) )
        except:
            curr_prob = self.ceiling
        
        return curr_prob
    