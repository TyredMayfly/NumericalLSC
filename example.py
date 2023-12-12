# -*- coding: utf-8 -*-
"""
@author: Thomas de Bruin
"""
# Import the wrapper class
from algorithm.algorithm_class import AllAlgorithmsClass

# Initialize the algorithm
numerical_algorithm = AllAlgorithmsClass()

# Run the test to ensure correct implementation
numerical_algorithm.test_all_numerical()

#%% Single LSC

# Create an instance of the class, see the class or the documentation for all possible parameters. 
algorithm_instance = AllAlgorithmsClass(
    spectrum_start=300,  # Change starting wavelength to 300 nm
    spectrum_end=1400,   # Change ending wavelength to 1400 nm
    width=5.0,           # Change width to 5.0 cm
    height=10.0,         # Change height to 10.0 cm
    quantum_yield_unity=True  # Set quantum yield to unity
)

luminophore = "Lee et al., 2023_AIGS"
concentration = 30

pce, avt, cri = algorithm_instance.numerical_single_wrapper(luminophore, concentration)

