# coupledOscillators
A set of functions and classes for fitting angle-resolved reflectivity data with a coupled oscillator model for polaritonics


Best Practice:
1) Measure angle resolved reflectivity of a sample using VASE
2) use loadRef to load the .dat file
3) use bumpFit class to extract the transition energies for each dip in the reflectivity spectrum
4) use fit to model the dispersion of the transistion energies using a coupled oscillator model
