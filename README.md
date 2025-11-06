# UltracoldAtomTransport
Simulations for atom loading &amp; transport in a moving optical lattice. 

## Motivation
These simulations were originally written to aid in the design of a transport module for an optical lattice clock. In principle, these codes can be used to study the loading &amp; transport of tightly-confined atoms in optical lattices for a wide range of experimental conditions. In particular, these simulations allows for easy customization of atomic species, laser wavelength, lens design, and ramping profiles. 

## Important Note
There are two versions of the main source code: (1) montepython (written in Python) and (2) julian (written in Julia). This provides wider user accessibility, and also serves as a performance benchmark for Python+Numba JIT compiler vs. Julia. 

## Usage
Both montepython and julian follow a similar workflow:
  - Create an atomic species: mass of atom &amp; GS --> ES transitions.
  - Initialize the loading position &amp; velocity for the atoms.
  - Create the optical system
    (an array contaning the postion and focal length of each lens element).
  - Define a Gaussian beam object
    (each beam forming the optical lattice is specified by the following parameters: initial position, propagation direction, wavelength, initial beam width, beam power)
  - Define a ramp profile object for the optical lattice
    (current ramp profiles: "None", "Linear", "Minimum Jerk").
  - Define simulation parameters (time step, simulation time, number of atoms, initial cloud position distribution, temperature).
  - Run the simulation file (MCsim.py for montepython, simfile.jl for julian) --> returns the positions &amp; velocities of each atom at each time step of the simulation.

## Pending Updates
  - Focus-tunable lenses (implementing thin lenses with time-dependent focal length).
  - New beam profiles (Higher-order TEM Gaussian beams, Bessel beams, etc.) and new ramp profiles.
  - Simulated Landau-Zener tunneling due to suppressed trap depth in accelerated lattices.
