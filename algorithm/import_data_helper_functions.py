# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:34:52 2022

@author: Bruin073
@name: data_import_helper_functions

"""
#%% Import relevant libraries and classes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.set_loglevel("info") 

from scipy.interpolate import interp1d # Used for interpolation of imported spectra

# Set relative path
import os
current_dir = os.getcwd()


#%% Import light distribution functions

def import_spectrum(spectrum_type, 
                    name=None, 
                    spectrum_start=280, 
                    spectrum_end=1500, 
                    interval=5, 
                    plotting=False):
    """
    Imports and interpolates different types of spectra: emission, absorption, or AM1.5G photon flux.

    Parameters:
    - spectrum_type (str): Type of the spectrum. Can be "emission", "absorption", or "AM1.5G".
    - name (str): Name of the luminophore, used for emission and absorption spectra.
    - spectrum_start (int): Starting wavelength for interpolation.
    - spectrum_end (int): Ending wavelength for interpolation.
    - interval (int): Wavelength interval for interpolation.
    - plotting (bool): If True, plots the interpolated spectrum.

    Returns:
    - df_interp (pd.DataFrame): Interpolated spectrum dataframe.
    """

    # Define wavelength range
    wavelength_range = np.arange(spectrum_start, spectrum_end + interval, interval)
    
    # Determine file path and column names based on spectrum type
    if spectrum_type == "emission":
        if name is None:
            raise ValueError("For emission and absorption spectra, 'name' parameter must be provided.")
        file_path = os.path.join(current_dir, "Data", "Luminophores", name, name + "_Ems.csv")
        column_name = 'emission'
    elif spectrum_type == "absorption":
        if name is None:
            raise ValueError("For emission and absorption spectra, 'name' parameter must be provided.")
        file_path = os.path.join(current_dir, "Data", "Luminophores", name, name + "_Abs.csv")
        column_name = 'absorption'
    elif spectrum_type == "AM1.5G":
        file_path = os.path.join(current_dir, "Data", "photon_flux_cm2_s.csv")
        column_name = 'spectrum'
    else:
        raise ValueError("Invalid spectrum_type provided.")

    # Load the data
    df = pd.read_csv(file_path, sep=',', decimal='.')

    # Interpolate
    interp = interp1d(df.wavelength,
                      df[column_name],
                      bounds_error=False,
                      fill_value=0.0)
    
    # For AM1.5G data, multiply by conversion factor and interval
    if spectrum_type == "AM1.5G":
        _cm2_to_m2 = 10**4
        df_interp = pd.DataFrame(data=[wavelength_range, interp(wavelength_range) * _cm2_to_m2 * interval]).T
    else:
        df_interp = pd.DataFrame(data=[wavelength_range, interp(wavelength_range)]).T

    # Set column names
    df_interp.columns = ['wavelength', column_name]
    
    # PVtrace compatibility: force all values to be non-negative
    df_interp[df_interp < 0] = 0
        
    # Optionally plot the data
    if plotting:
        plt.rcParams['figure.dpi'] = 200
        plt.plot(df_interp.wavelength, df_interp[column_name])
        plt.show()

    return df_interp


def import_quantum_yield(name=None):
    """
    Imports the quantum yield for a given luminophore.
    
    Args:
        name (str): Name of the luminophore. If None, it defaults to the directory structure.
        
    Returns:
        float: Quantum yield of the luminophore, or None if unsuccessful.
    """
    
    if not name:
        raise ValueError("A luminophore name must be provided.")
    
    filepath = os.path.join(current_dir, "Data", "Luminophores", name, "QY.csv")
    
    try:
        quantum_yield = pd.read_csv(filepath, header=None).iloc[0, 0]
        return float(quantum_yield)
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Error importing quantum yield for {name}: {e}")
        return None






