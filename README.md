# UltracoldAtomTransport
Simulations for atom loading &amp; transport in a moving optical lattice. 

The main simulation functions are defined in sys_funcs.py. The basic layout is as follows: 

Creating a Gaussian beam: 
  - prop_matrix() takes a position z along the optical axis and an array of lens elements (each element contains position of lens + focal length) and returns the ABCD matrix at z.
  - q() calculates the complex beam parameter corresponding to prop_matrix(), and beam_width() calculates the beam width from q().
  - I_field() returns the intensity profile of a Gaussian beam at a position z along the optical axis using beam_width().

Defining a lattice ramping profile:
   - z_latt() returns the ramping profile of the lattice (the position of optical dipole trap minimum as a function of time).
   - v_latt() returns the corresponding velocity profile

Getting optical dipole trap (ODT) info:
  - trap_info() takes in the wavelength of the incident beam and the transition wavelength + linewidth associated with a particular atomic transition and returns the detuning, dipole matrix        element, and (scalar) atomic polarizability. 

Lattice functions: 
  - I_lattice() takes two Gaussian beams (for each beam, one must specify an initial position r0, propagation direction prop_dir, initial beam waist w_0, and beam power P) and returns the          corresponding intensity profile due to their interference. The phase is controlled by the lattice ramping profile z_latt().
  - U_lattice() uses I_lattice() and the atomic polarizability ot calculate the optical dipole potential of the lattice.
  - F_lattice() calculates the minus the gradient of U_lattice().

Simulation functions:
  - MC_step() returns one step of the Monte Carlo simulation. It takes an initial position + velocity and calculates the position + velocity at a later time step according to Newton's Laws         (the force is given by F_lattice())
  - one_atom_sim() runs MC_step() for a single atom at each time step in the simulation window.
  - run_MC_sim() performs one_atom_sim() for N_atoms in parallel using Numba's prange() function. 


The simulation function run_MC_sim() is treated as a wrapper function, which is passed through the function MCsim() in the MCtransport class, which is defined in MCclass.py. 
MCsim() takes a params dictionary containing the relevant system/simulation parameters, and returns the positions/velocities of the atoms at each time step. params is defined in MCsim.py. The
results of the simulation are stored in a .pkl file, which can be loaded and plotted in plots.py
