class Sim_Params:
    """
    Simulation params hold information about system setting for simulation
    - timestep - 5 min per timestep
    - length - 1 mile per unit length
    """
    
    def __init__(self, time_steps, x_length, y_length):
        
        self.time_steps = time_steps
        self.x_length = x_length
        self.y_length = y_length
