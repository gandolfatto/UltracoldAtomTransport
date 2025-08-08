from sys_funcs import *
from typing import Dict, Any


class MCtransport:

    def __init__(self, params: Dict[str, Any]) -> None: 

        self.pos_load = params["pos load"]
        self.vel_load = params["vel load"]

        self.lens_elements = params["lens elements"]
        
        self.r0_1 = params["r0_1"]
        self.r0_2 = params["r0_2"]
        self.prop_dir1 = params["prop dir1"]
        self.prop_dir2 = params["prop dir2"]
        self.wavelength = params["wavelength"]
        self.w0_1 = params["w0_1"]
        self.w0_2 = params["w0_2"]
        self.P1 = params["P1"]
        self.P2 = params["P2"]

        self.d_max = params["d max"]
        self.t_max = params["t max"]

        self.transitions = params["transitions"]

        self.dt = params["dt"]
        self.t_sim = params["t sim"]
        self.N_atoms = params["N atoms"]
        self.sigma_x = params["sigma x"]
        self.T = params["T"]                                                        
        self.sigma_v = np.sqrt(kb*self.T/m)


    def MCsim(self):
        return run_MC_sim(self.pos_load, self.vel_load, 
                          self.lens_elements, 
                          self.r0_1, self.r0_2, self.prop_dir1, self.prop_dir2, self.wavelength, self.w0_1, self.w0_2, self.P1, self.P2, 
                          self.d_max, self.t_max, 
                          self.transitions, 
                          self.dt, self.t_sim, self.N_atoms, self.sigma_x, self.sigma_v)
