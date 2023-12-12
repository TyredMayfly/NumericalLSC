# Import relevant functions
from .numerical_algorithms import (
   numerical_single, numerical_double, numerical_triple, numerical_multiple_in_one
)

import time  # To measure the execution time of algorithms


class AllAlgorithmsClass:
    """
    A class designed to encapsulate different algorithms for numerical simulations.
    """
    
    # Default parameters for simulation
    DEFAULT_PARAMS = {
        "spectrum_start": 280,          # Starting wavelength of the spectrum in nm
        "spectrum_end": 1500,           # Ending wavelength of the spectrum in nm
        "interval": 5,                  # Interval for spectrum calculations in nm
        "width": 4.0,                  # Width of LSC in cm
        "height": 8.0,                 # Height of LSC in cm
        "thickness": 0.5,               # Thickness of LSC in cm
        "refractive_index_wg": 1.5,     # Refractive index of waveguide
        "refractive_index_air": 1.0,    # Refractive index of air
        "quantum_yield_unity": False,   # A simulation option for unity quantum yield
        "waveguide_absorption_off": False,  # A simulation option to turn off waveguide absorption
        "print_options": True
    }

    def __init__(self, **kwargs):
        """
        Initialize the AllAlgorithmsClass with default parameters and update with given arguments.
        
        :param print_options: Boolean, if True then simulation options will be printed
        :param **kwargs: Any extra arguments to override the default parameters
        """
        
        # Copy default parameters and update with any given arguments
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(kwargs)
        
        # Create a dictionary of parameters for easy access
        self.parameter_dict = self.params

    def _print_sim_options(self):
        """
        Print the current simulation options set in the parameter dictionary.
        """
        # Print a header
        print("\n==================== Simulation Parameters ====================")
        
        # Print all parameters in a formatted way
        max_key_length = max([len(key) for key in self.params.keys()])
        for key, value in self.params.items():
            print(f"{key.ljust(max_key_length)} : {value}")
    
        # Print specific simulation options with a line break for clarity
        if self.params["quantum_yield_unity"]:
            print("\n[INFO] quantum_yield_unity is active.")
        
        if self.params["waveguide_absorption_off"]:
            print("\n[INFO] waveguide_absorption_off is turned off.")
        
        # Check if both quantum_yield_unity and waveguide_absorption_off are off and print a specific message
        if not self.params["quantum_yield_unity"] and not self.params["waveguide_absorption_off"]:
            print("\n[NOTE] Both quantum_yield_unity and waveguide_absorption_off are turned OFF.")
        
        print("\n===============================================================\n")

    def _execute_algorithm(self, algorithm_function, *args):
           """
           A helper function to execute a given algorithm and measure its execution time.
       
           :param algorithm_function: The function (algorithm) to be executed
           :param *args: Arguments required by the algorithm
           :return: Results returned by the algorithm_function
           """
           
           start_time = time.time()  # Record the starting time
           
           # Check the print_options value from the params dictionary
           if self.params.get("print_options"):
               self._print_sim_options()
    
           print(f"Executing algorithm: {algorithm_function.__name__}")
    
           results = algorithm_function(*args, self.parameter_dict)
           
           end_time = time.time()  # Record the ending time
    
           # Calculate and print the duration of execution
           duration = end_time - start_time
           print(f"Algorithm {algorithm_function.__name__} executed in {duration:.2f} seconds.")
           
           return results


    # The following methods are wrappers around the imported algorithms, making it easy to use them with the current parameter settings.
    # Each method takes in specific parameters required by the algorithm it wraps around.
        
    def numerical_single_wrapper(self, luminophore_1, concentration_1):
        return self._execute_algorithm(numerical_single, luminophore_1, concentration_1)

    def numerical_double_wrapper(self, luminophore_1, concentration_1, luminophore_2, concentration_2):
        return self._execute_algorithm(numerical_double, luminophore_1, concentration_1, luminophore_2, concentration_2)

    def numerical_triple_wrapper(self, luminophore_1, concentration_1, luminophore_2, concentration_2, luminophore_3, concentration_3):
        return self._execute_algorithm(numerical_triple, luminophore_1, concentration_1, luminophore_2, concentration_2, luminophore_3, concentration_3)

    # This can potentially be run with a single luminophore, therefore standard arguments with concentration = 0 are given. 
    def numerical_multiple_in_one_wrapper(self, luminophore_1, concentration_1, luminophore_2="Lee et al., 2023_AIGS", concentration_2=0.0, luminophore_3="Lee et al., 2023_AIGS", concentration_3=0.0):
        return self._execute_algorithm(numerical_multiple_in_one, luminophore_1, concentration_1, luminophore_2, concentration_2, luminophore_3, concentration_3)

    def test_all_numerical(self):
        decimals = 5
    
        original_params = self.params.copy()

    
        # Updating the parameters for validation
        self.params.update({
            "spectrum_start": 280,
            "spectrum_end": 1500,
            "interval": 5,
            "width": 4.0,
            "height": 8.0,
            "thickness": 0.5,
            "refractive_index_wg": 1.5,
            "refractive_index_air": 1.0,
            "quantum_yield_unity": False,
            "waveguide_absorption_off": False
        })
    
    
        self.parameter_dict = self.params

        # Test the four different algorithms with chosen parameters and materials
        # Results are in the form of (PCE, AVT, CRI) for each algorithm       
        pce_n_single, avt_n_single, cri_n_single = self.numerical_single_wrapper("Lee et al., 2023_AIGS", 30)
        pce_n_double, avt_n_double, cri_n_double = self.numerical_double_wrapper("Lee et al., 2023_AIGS", 5, "Wei et al., 2022_MN-MQW-perov", 4)
        pce_n_triple, avt_n_triple, cri_n_triple = self.numerical_triple_wrapper("Lee et al., 2023_AIGS", 4, "Wei et al., 2022_MN-MQW-perov", 5, "Gong et al., 2022_Si-C-dots", 2)
        pce_n_mio, avt_n_mio, cri_n_mio = self.numerical_multiple_in_one_wrapper("Lee et al., 2023_AIGS", 4, "Wei et al., 2022_MN-MQW-perov", 5, "Gong et al., 2022_Si-C-dots", 3)

        self.params = original_params

        pce_single_validation = 0.892263404
        avt_single_validation = 79.1835000
        cri_single_validation = 36.12177817
         
        pce_double_validation = 1.04000455
        avt_double_validation = 81.6904931
        cri_double_validation = 75.63497506
          
        pce_triple_validation = 1.33719948
        avt_triple_validation = 66.74691209
        cri_triple_validation = 68.1762545
        
        pce_mio_validation = 0.772265840
        avt_mio_validation = 77.8145331624
        cri_mio_validation = 61.99240563

        def print_validation_results(name, pce_value, avt_value, cri_value, pce_validation, avt_validation, cri_validation, decimals):
            """
            A helper function to print validation results in a formatted manner.
            """
            print(f"{name} LSC validation ({decimals} decimals):\n"
                  f"PCE ratio: {round(pce_value/pce_validation, decimals)}\n"
                  f"AVT ratio: {round(avt_value/avt_validation, decimals)}\n"
                  f"CRI ratio: {round(cri_value/cri_validation, decimals)}\n")
            print('-'*50)  # Print separator for better visual distinction
        
        print("\n", "#"*20, " VALIDATION RESULTS ", "#"*20, "\n")
        
        # Use the helper function to print results for each algorithm
        print_validation_results("Single", pce_n_single, avt_n_single, cri_n_single, pce_single_validation, avt_single_validation, cri_single_validation, decimals)
        print_validation_results("Double", pce_n_double, avt_n_double, cri_n_double, pce_double_validation, avt_double_validation, cri_double_validation, decimals)
        print_validation_results("Triple", pce_n_triple, avt_n_triple, cri_n_triple, pce_triple_validation, avt_triple_validation, cri_triple_validation, decimals)
        print_validation_results("Multiple in one", pce_n_mio, avt_n_mio, cri_n_mio, pce_mio_validation, avt_mio_validation, cri_mio_validation, decimals)

