from MCclass import *

import pickle
import time



### lens elements
lens_eles = np.array([
    [1.6*cm,  18*cm],
    [51.6*cm, 18*cm]
])

### transition info: [transition wavelength, linewidth]
transitions = np.array([
                        [399*nm, 2*np.pi*29.1e6], 
                        [556*nm, 2*np.pi*181.1e3]
])


params = {  # configuration parameters
            "pos load": np.array([0, 0, 11.6*cm]),
            "vel load": 0.,
            "lens elements": lens_eles,


            # beam parameters
            "r0_1": 0*cm,
            "r0_2": 53.2*cm, 
            "prop dir1": 1.,
            "prop dir2": -1., 
            "wavelength": 1064*nm, 
            "w0_1": 362*um,
            "w0_2": 362*um,
            "P1": 200,
            "P2": 200,


            # ramp/lattice parameters
            "d max": 30*cm, 
            "t max": 50*ms, 


            # transition parameters
            "transitions": transitions,


            # simulation parameters
            "dt": 1e-5,  
            "t sim": 1*ms, 
            "N atoms": 8, 
            "sigma x": np.array([0.02*mm, 0.02*mm, 0.02*mm]),
            "T": 1e-6
}



# SIMULATION
if __name__ == '__main__':
    start = time.perf_counter()

    mot = MCtransport(params)
    ts, rs, vs = mot.MCsim()

    stop = time.perf_counter()

    print(f"The simulation for {params["N atoms"]} atoms took {round(stop - start, 1)} s.")

    results = {"ts": ts,
               "rs": rs,
               "vs": vs
               }

    suffix = 'transport_sim'
    with open(f'parameters{suffix}.pkl', 'wb') as outp:
        pickle.dump(params, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'results{suffix}.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)