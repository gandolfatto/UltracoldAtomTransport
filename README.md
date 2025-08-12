# UltracoldAtomTransport
Simulations for atom loading &amp; transport in a moving optical lattice. 

## Motivation
These simulations were originally written to aid in the design of the transport module for an optical lattice clock. In principle, they can be used to study the loading/transport of tightly-confined atoms in optical lattices for a wide range of experimental conditions. In particular, these simulations allows for easy customization of atomic species, laser wavelength, lens design, and ramping profiles. 

## Important Note
There are two versions of the main source code: (1) montepython (written in Python) and (2) kingjulian (written in Julia). This provides wider user accessibility, and also serves as a performance benchmark for Python+Numba JIT compilation vs. Julia. 

## Usage
Both montepython and kingjulian follow a similar workflow:
  - Initialize the loading position + velocity for the atoms.
  - Create the optical system (an array contaning the postion and focal length of each lens element).
  - Define Gaussian beam parameters (each beam forming the optical lattice takes the following parameters: initial position, propagation direction, wavelength, initial beam width, beam power)
  - Define a ramping profile for the optical lattice (current default: "minimum jerk trajectory").
  - Create an atomic species: mass of atom + relevant GS --> Excited state transitions (transitions encoded in an array, which contains transition wavelength and linewidth).
  - Define simulation parameters (time step, simulation time, number of atoms, initial cloud position distribution, temperature).
  - Perform the simulation with run_MC_sim() --> returns the positions + velocities of each atom at each time step of the simulation.

## Pending Updates
### Major
  - Focus-tunable lenses (i.e, implementing lenses with time-dependent focal length).
  - New beam profiles (Higher-order TEM Gaussian beams, Bessel beams, etc.).
  - New ramp profiles.
  - Simulated Landau-Zener tunneling due to suppressed trap depth in accelerated lattices.
### Minor
  - Simulated parametric heating due to intensity noise fluctuations along transverse axis.
  - Simulated loss due to spontaneuous emission (less important in far-off resonant traps)
