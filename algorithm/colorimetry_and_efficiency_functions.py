# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:41:03 2021

@author: Bruin073
"""

import pandas as pd # Used for data management
import numpy as np # Used for natural constants number management
from scipy import interpolate # Used for interpolation of imported spectra

import os # Used for specifying local directory
current_dir = os.getcwd() # Specify local directory

from .import_data_helper_functions import import_spectrum

#%%


def compute_power_spectrum(wavelengths, photon_flux, intensity_factor, bin_interval=5):
    """
    Computes the power spectrum from a given photon flux.
    
    Parameters:
    - wavelengths (np.array): Array of wavelengths for the photons.
    - photon_flux (np.array): Corresponding flux for each wavelength.
    - intensity_factor (float): A multiplier for the power calculation.
    - bin_interval (int, optional): Interval for binning the wavelengths. Default is 5.
    
    Returns:
    - DataFrame: Contains photon_count, average_wavelength, and power for each binned wavelength.
    """
    
    # Constants
    light_spectrum_start = 280
    light_spectrum_end = 4000
    speed_of_light = 3e8  # in m/s
    planck_constant = 6.626e-34  # in J*s
    
    # Check if binning is possible with the given interval
    assert (light_spectrum_end - light_spectrum_start) % bin_interval == 0, "Invalid bin size for the given spectrum range."
    
    # Create bins
    bins = np.arange(light_spectrum_start, light_spectrum_end + bin_interval, bin_interval)
    
    # Bin data
    indices = np.digitize(wavelengths, bins)
    
    # Calculate photon counts, average wavelengths, and power
    photon_counts, _ = np.histogram(wavelengths, bins=bins)
    average_wavelengths = np.bincount(indices, weights=wavelengths) / np.maximum(1, photon_counts) * 10**-9
    power = photon_counts * planck_constant * speed_of_light / average_wavelengths * intensity_factor
    
    # Construct the result DataFrame
    result_df = pd.DataFrame({
        'photon_count': photon_counts,
        'average_wavelength': average_wavelengths,
        'power': power
    }, index=bins[:-1])
    
    result_df.fillna(0, inplace=True)
    
    return result_df

def normalize_photon_flux(df, ray_trace_multiplier, spectrum_start=300, spectrum_end=1500, interval=5):
    # Adjusted bin range to ensure the bin centers start at spectrum_start
    bin_edges = np.arange(spectrum_start - (interval / 2), spectrum_end + (interval / 2), interval)
    
    # Use np.histogram to calculate photon counts for each bin
    photon_counts, _ = np.histogram(df['wavelength'], bins=bin_edges)
    
    # Calculate the photon flux
    photon_flux = photon_counts * ray_trace_multiplier
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create a 2D numpy array with bin centers and photon flux
    photon_array = np.column_stack((bin_centers, photon_flux))
    
    return photon_array


def calculate_cri(wavelengths_of_spectrum, transmitted_spectrum, interval=5):
    
    """
    Calculate the Color Rendering Index (CRI) for a given spectrum.
    
    Parameters:
    - wavelengths_of_spectrum (np.array): Wavelengths of the transmitted spectrum.
    - transmitted_spectrum (np.array): Intensities for the transmitted spectrum at the given wavelengths.
    - interval (int, optional): The binning interval for the spectrum. Default is 5.
    - ray_number_multiplier (float, optional): A factor to multiply the power spectrum if needed. Default is 1.
    
    Returns:
    - interp1d object: Interpolated transmitted spectrum.
    """
            
    spectrum_start = 300
    spectrum_end = 900
    
    # Ensure the given interval is compatible with the spectrum range
    assert(spectrum_end % interval == 0), "Invalid bin size for the given spectrum range."
    
    new_range = np.arange(spectrum_start, spectrum_end + interval, interval)  # Define the datapoints for the spectra 
    
    # Compute the power spectrum and interpolate if ray number multiplier is not 1
    transmitted_df = pd.DataFrame({'wavelength': wavelengths_of_spectrum, 'spectrum': transmitted_spectrum})
    transmitted_df_interpolated = interpolate.interp1d(transmitted_df['wavelength'], transmitted_df['spectrum'], bounds_error=False, fill_value=0.0)
    
    transmission = pd.DataFrame({'wavelength': new_range, 'spectrum': transmitted_df_interpolated(new_range)})
    
    # Import am1.5g energy flux (W/m^2/nm)
    light_spectrum = pd.read_csv(current_dir+"/Data/am1_5_energy_flux.csv", delimiter=";")
    light_spectrum_interp = interpolate.interp1d(light_spectrum.wavelength, light_spectrum.spectrum, bounds_error=False, fill_value=0.0)
    light_spectrum = pd.DataFrame({'wavelength': new_range, 'spectrum': light_spectrum_interp(new_range)})
    
    # Import and interpolate colorimetric data
    color_matching_functions_import = pd.read_csv(current_dir+"/Data/color_matching_functions.csv", delimiter=";", decimal=".")
    color_interp_funcs = {
        'x_bar': interpolate.interp1d(color_matching_functions_import.wavelength_nm, color_matching_functions_import.x_bar, bounds_error=False, fill_value=0.0),
        'y_bar': interpolate.interp1d(color_matching_functions_import.wavelength_nm, color_matching_functions_import.y_bar, bounds_error=False, fill_value=0.0),
        'z_bar': interpolate.interp1d(color_matching_functions_import.wavelength_nm, color_matching_functions_import.z_bar, bounds_error=False, fill_value=0.0)
    }
    color_matching_functions = pd.DataFrame({
        'wavelength': new_range, 
        'x_bar': color_interp_funcs['x_bar'](new_range),
        'y_bar': color_interp_funcs['y_bar'](new_range),
        'z_bar': color_interp_funcs['z_bar'](new_range)
    })
    
    reference_colors_import = pd.read_csv(current_dir+"/Data/reference_colors_8_colors.csv", delimiter=";", decimal=",")
    reference_colors_list = [new_range]
       
    for columnName, columnData in reference_colors_import.items():
        if columnName != "wavelength_nm":
            test_color_interp = interpolate.interp1d(reference_colors_import.wavelength_nm, columnData, bounds_error=False, fill_value=0.0)
            reference_colors_list.append(test_color_interp(new_range))
    reference_colors = pd.DataFrame(data=np.array(reference_colors_list).T, columns=reference_colors_import.columns)
    
    # Normalization and other calculations
    y_normalization_factor = sum(color_matching_functions.y_bar * transmission.spectrum)
    y_normalization_factor_reference = sum(color_matching_functions.y_bar * light_spectrum.spectrum) 
    y_normalized_reference_spectrum = light_spectrum.spectrum / y_normalization_factor_reference * 100
    
    if y_normalization_factor < 10**-5:
        print("Transmission too little to calculate CRI, setting at 0.0")
       # return 0.0
    
       # Calculate tristimulus values for transmitted light
    X = sum(transmission.spectrum * color_matching_functions.x_bar) / y_normalization_factor * 100
    Y = sum(transmission.spectrum * color_matching_functions.y_bar) / y_normalization_factor * 100
    Z = sum(transmission.spectrum * color_matching_functions.z_bar) / y_normalization_factor * 100
    
    # Calculate x, y coordinates
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    
    # Calculate 1976 u,v coordinates for transmitted light
    u_transmitted = 4 * X / (X + 15 * Y + 3 * Z)
    v_transmitted = 9 * Y / (X + 15 * Y + 3 * Z)
    
    # Load data for planckian locus temperatures
    planckian_locus_temperatures = pd.read_csv(current_dir + "/Data/planckian_locus_temperatures.csv", delimiter=";")
    planckian_locus_temperatures['dist_1960'] = ((u_transmitted - planckian_locus_temperatures['u_1960'])**2 + 
                                                (v_transmitted - planckian_locus_temperatures['v_1960'])**2)**0.5
    planckian_distance = min(planckian_locus_temperatures['dist_1960'])
    planckian_limit = 0.0025
    
    cri_calc_index = ["X",
                  "Y",
                  "Z",
                  "x",
                  "y",
                  "u",
                  "v",
                  "(Y/Yn)",
                  "L*",
                  "u*",
                  "v*"]
    
    
    # Initialize DataFrame for CRI calculations reference
    cri_calc_reference = pd.DataFrame(index=cri_calc_index, columns=reference_colors.columns)
    cri_calc_reference.drop("wavelength_nm", axis=1, inplace=True)
    
    # Calculate values for the reference white and the test color
    for columnName, columnData in reference_colors.items():
        if columnName == "wavelength_nm":
            continue
    
        # Calculate tristimulus values for reference
        X_ref = sum(color_matching_functions.x_bar * y_normalized_reference_spectrum * columnData)
        Y_ref = sum(color_matching_functions.y_bar * y_normalized_reference_spectrum * columnData)
        Z_ref = sum(color_matching_functions.z_bar * y_normalized_reference_spectrum * columnData)
    
        # Calculate x, y coordinates for reference
        x_ref = X_ref / (X_ref + Y_ref + Z_ref)
        y_ref = Y_ref / (X_ref + Y_ref + Z_ref)
    
        # Calculate 1976 u,v coordinates for reference
        u_ref = 4 * x_ref / (-2 * x_ref + 12 * y_ref + 3)
        v_ref = 6 * y_ref / (-2 * x_ref + 12 * y_ref + 3)
    
        c_ref = (4-u_ref-10*v_ref)/v_ref
        d_ref = (1.708*v_ref+0.404-1.481*u_ref)/v_ref
    
        # Store computed values
        cri_calc_reference.at['X', columnName] = X_ref
        cri_calc_reference.at['Y', columnName] = Y_ref
        cri_calc_reference.at['Z', columnName] = Z_ref
        cri_calc_reference.at['x', columnName] = x_ref
        cri_calc_reference.at['y', columnName] = y_ref
        cri_calc_reference.at['u', columnName] = u_ref
        cri_calc_reference.at['v', columnName] = v_ref
        cri_calc_reference.at['cref', columnName] = c_ref
        cri_calc_reference.at['dref', columnName] = d_ref
    
        # Intermediate (Y/Y_n) value
        Y_Yn_ratio = Y_ref / 100
        if Y_Yn_ratio > (6/29)**3:
            Y_Yn_ratio = Y_Yn_ratio**(1/3)
        else:
            Y_Yn_ratio = (841/108) * Y_Yn_ratio + 4/29
        cri_calc_reference.at["(Y/Yn)", columnName] = Y_Yn_ratio
    
        # Calculate variables for lightness and CRI calculations
        if planckian_distance < planckian_limit:
            L = 116 * Y_Yn_ratio - 16
        else:
            L = 25 * Y_ref**(1/3) - 17
    
        u = 13 * L * (u_ref - cri_calc_reference.at['u', 'White'])
        v = 13 * L * (v_ref - cri_calc_reference.at['v', 'White'])
    
        cri_calc_reference.at["L*", columnName] = L
        cri_calc_reference.at["u*", columnName] = u
        cri_calc_reference.at["v*", columnName] = v
        
    # Initialize DataFrame for CRI calculations with test spectrum
    cri_calc_test_spectrum = pd.DataFrame(index=cri_calc_index, columns=reference_colors.columns)
    cri_calc_test_spectrum.drop("wavelength_nm", axis=1, inplace=True)
    
    for columnName, columnData in reference_colors.items():
        if columnName == "wavelength_nm":
            continue
    
        # Calculate tristimulus values
        X_test = sum(transmission.spectrum * color_matching_functions.x_bar * columnData) / y_normalization_factor * 100
        Y_test = sum(transmission.spectrum * color_matching_functions.y_bar * columnData) / y_normalization_factor * 100
        Z_test = sum(transmission.spectrum * color_matching_functions.z_bar * columnData) / y_normalization_factor * 100
        
        # Calculate x, y coordinates
        x_test = X_test / (X_test + Y_test + Z_test)
        y_test = Y_test / (X_test + Y_test + Z_test)
    
        # Calculate 1976 u,v coordinates
        u_test = 4 * x_test / (-2 * x_test + 12 * y_test + 3)
        v_test = 6 * y_test / (-2 * x_test + 12 * y_test + 3)
    
        c_test = (4-u_test-10*v_test)/v_test
        d_test = (1.708*v_test+0.404-1.481*u_test)/v_test
    
    
        # Store computed values
        cri_calc_test_spectrum.at['X', columnName] = X_test
        cri_calc_test_spectrum.at['Y', columnName] = Y_test
        cri_calc_test_spectrum.at['Z', columnName] = Z_test
        cri_calc_test_spectrum.at['x', columnName] = x_test
        cri_calc_test_spectrum.at['y', columnName] = y_test
        cri_calc_test_spectrum.at['u', columnName] = u_test
        cri_calc_test_spectrum.at['v', columnName] = v_test
        cri_calc_test_spectrum.at['ctest', columnName] = c_test
        cri_calc_test_spectrum.at['dtest', columnName] = d_test
    
        u_chrom_corr = (10.872 + 0.404 * cri_calc_reference.loc['cref', 'White'] / cri_calc_test_spectrum.loc['ctest', 'White'] * cri_calc_test_spectrum.loc['ctest', columnName] - 4 * cri_calc_reference.loc['dref', 'White'] / cri_calc_test_spectrum.loc['dtest', 'White'] * cri_calc_test_spectrum.loc['dtest', columnName]) / (16.518 + 1.481 * cri_calc_reference.loc['cref', 'White'] / cri_calc_test_spectrum.loc['ctest', 'White'] * cri_calc_test_spectrum.loc['ctest', columnName] - cri_calc_reference.loc['dref', 'White'] / cri_calc_test_spectrum.loc['dtest', 'White'] * cri_calc_test_spectrum.loc['dtest', columnName])
        v_chrom_corr = 5.52 / (16.518 + 1.481 * cri_calc_reference.loc['cref', 'White'] / cri_calc_test_spectrum.loc['ctest', 'White'] * cri_calc_test_spectrum.loc['ctest', columnName] - cri_calc_reference.loc['dref', 'White'] / cri_calc_test_spectrum.loc['dtest', 'White'] * cri_calc_test_spectrum.loc['dtest', columnName])
    
        cri_calc_test_spectrum.at['u_chrom_corr', columnName] = u_chrom_corr
        cri_calc_test_spectrum.at['v_chrom_corr', columnName] = v_chrom_corr
    
        # Intermediate (Y/Y_n) value
        Y_Yn_ratio = Y_test / 100
        
        if Y_Yn_ratio > (6/29)**3:
            Y_Yn_ratio = Y_Yn_ratio**(1/3)
        else:
            Y_Yn_ratio = (841/108) * Y_Yn_ratio + 4/29
        cri_calc_test_spectrum.at["(Y/Yn)", columnName] = Y_Yn_ratio
    
        # Calculate variables for lightness and CRI calculations
        if planckian_distance < planckian_limit:
            L = 116 * Y_Yn_ratio - 16
            u = 13 * L * (u_test - cri_calc_test_spectrum.at['u', 'White'])
            v = 13 * L * (v_test - cri_calc_test_spectrum.at['v', 'White'])
        else:
            L = 25 * Y_test**(1/3) - 17
            u = 13 * L * (u_chrom_corr - cri_calc_test_spectrum.at['u_chrom_corr', 'White'])
            v = 13 * L * (v_chrom_corr - cri_calc_test_spectrum.at['v_chrom_corr', 'White'])
    
        cri_calc_test_spectrum.at["L*", columnName] = L
        cri_calc_test_spectrum.at["u*", columnName] = u
        cri_calc_test_spectrum.at["v*", columnName] = v
    
    # Calculate CRI
    cri_results = pd.DataFrame(index=["delta E interm.", "delta E", "CRI"], columns=reference_colors.columns)
    cri_results.drop("wavelength_nm", axis=1, inplace=True)
    
    cri_results.loc["delta E interm."] = (cri_calc_reference.loc['L*', :] - cri_calc_test_spectrum.loc['L*', :])**2 + \
                                         (cri_calc_reference.loc['u*', :] - cri_calc_test_spectrum.loc['u*', :])**2 + \
                                         (cri_calc_reference.loc['v*', :] - cri_calc_test_spectrum.loc['v*', :])**2
    
    cri_results.loc["delta E"] = cri_results.loc["delta E interm.", :].pow(1./2)
    cri_results.loc["CRI"] = 100 - 4.6 * cri_results.iloc[1, 1:9]  # Don't include the reference white to average the CRI.
    
    cri = round(cri_results.loc["CRI"].mean(), 6)
    if cri < 0.0:
        cri = 0.0
     
    return cri





#%%

def calculate_avt(wavelengths_of_spectrum, 
                  transmitted_spectrum, 
                  interval=5):
    """
    Calculate Average Visual Transmission (AVT) based on the transmitted rays.
    
    Parameters:
    - stacked_transmission: 2D array containing wavelengths and transmitted spectrum.
    - interval: Wavelength interval for calculations.
    - ray_number_multiplier: Multiplier to normalize to AM1.5G.
    
    Returns:
    - Average Visual Transmission (AVT) value.
    """
    
    spectrum_start = 360
    spectrum_end = 830
   
    # Define the datapoints for the spectra
    wavelengths = np.arange(spectrum_start, spectrum_end + interval, interval)
   
    # Import the AM1.5G photon flux data using the import_spectrum function
    light_spectrum = import_spectrum("AM1.5G", spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval, plotting=False)

    am1_5g_photon_flux = light_spectrum['spectrum']
   
    # Import and interpolate the photopic response data
    photopic_response_data = np.genfromtxt(current_dir + "/Data/Photopic_response.csv", delimiter=";", skip_header=1)
    photopic_wavelengths, photopic_values = photopic_response_data[:, 0], photopic_response_data[:, 1]
    photopic_interp = interpolate.interp1d(photopic_wavelengths, photopic_values, bounds_error=False, fill_value=0.0)
    photopic_response = photopic_interp(wavelengths)
   
    # Interpolate transmitted spectrum to match our defined wavelength range
    transmitted_interp = interpolate.interp1d(wavelengths_of_spectrum, transmitted_spectrum, bounds_error=False, fill_value=0.0)
    transmission = transmitted_interp(wavelengths)
   
    # Calculate AVT by taking the dot product of the photopic response with the transmission, normalized by its dot product with the AM1.5G spectrum
    numerator = np.dot(photopic_response, transmission)
    denominator = np.dot(photopic_response, am1_5g_photon_flux)
    
    avt = numerator / denominator * 100
    
    return avt



# Constants dictionary
CONSTANTS = {
    'q': 1.6022e-19,  # Elementary charge in Coulombs
    'k_b': 1.38064852e-23,  # Boltzmann constant in m^2 kg s^-2 K^-1
    'temp': 298  # Standard room temperature in Kelvin
}
# Thermal voltage calculated from the constants
CONSTANTS['v_t'] = CONSTANTS['k_b'] * CONSTANTS['temp'] / CONSTANTS['q']


def calculate_photon_generated_current_density(eqe_wavelength, eqe_spectrum, photons_wavelength, photons_flux):
    """
    Calculate the photon-generated current density based on EQE data and incident photons.
    
    Parameters:
    - eqe_wavelength: Wavelength data for the EQE
    - eqe_spectrum: EQE values corresponding to eqe_wavelength
    - photons_wavelength: Wavelength data for the incident photons
    - photons_flux: Flux of incident photons corresponding to photons_wavelength

    Returns:
    - Photon-generated current density
    """
    
    # Interpolate the EQE data to match the wavelength range of the photons data
    interp = interpolate.interp1d(eqe_wavelength, eqe_spectrum, bounds_error=False, fill_value=0.0)
    interpolated_spectrum = interp(photons_wavelength)

    # Calculate and return the photon-generated current density
    current_density = CONSTANTS['q'] * np.sum(interpolated_spectrum * photons_flux)
    return current_density


def calculate_pce(side_incident_photons_wavelength,
                  side_incident_photons_flux,
                  spectrum_start=280,
                  spectrum_end=1500,
                  interval=5,
                  pv_cell_name='generic_Si',
				  r_sh=0.15,
				  r_s=0.00015,
                  diode_n = 1.2):
    """
    Calculate the power conversion efficiency (PCE) of a PV cell based on the provided EQE data and 
    incident photons.
    
    Parameters:
    - side_incident_photons_wavelength: Wavelength data for the side incident photons nm
    - side_incident_photons_flux: Flux of side incident photons corresponding to side_incident_photons_wavelength #/m^2/s
    - spectrum_start: Starting wavelength for the range
    - spectrum_end: Ending wavelength for the range
    - interval: Interval for the wavelength range
    - pv_cell_name: Name of the PV cell (currently only supports 'generic_Si')
    - r_sh: Shunt resistance Ohm-m^2
    - r_s: Series resistance Ohm-m^2
    - dionde_n: diode ideality factor

    Returns:
    - Power conversion efficiency (PCE) value in %
    """
    
    # Load EQE data for the specified PV cell
    if pv_cell_name == 'generic_Si':
        eqe_data = np.genfromtxt(os.path.join(current_dir, "Data/generic_Si.csv"), delimiter=";", skip_header=1)
        j_0_pv_cell = 3.25E-07  # Saturation current density for the PV cell
    else:
        raise ValueError(f"Unsupported PV cell type: {pv_cell_name}")

    # Split EQE data into wavelength and spectrum
    eqe_wavelength, eqe_spectrum = eqe_data[:, 0], eqe_data[:, 1]

    # Calculate the photon-generated current density
    photon_current_density = calculate_photon_generated_current_density(eqe_wavelength,
                                                                        eqe_spectrum, 
                                                                        side_incident_photons_wavelength,
                                                                        side_incident_photons_flux)
        
    # Handle cases where calculated current density is too low or invalid
    if photon_current_density < 10**-2 or np.isnan(photon_current_density):
        print('Error with calculating PCE, current is too low or nan, setting PCE at 0.0%')
        return 0.0

    # Define the voltage and current ranges for the PV cell performance
    voltage_range = np.arange(0.0, 1.01, 0.01)
    current_range = np.arange(0.0, 1.01, 0.01) * photon_current_density
    
    # Arrays for the transcendental equation calculation
    transcendental_current = np.zeros((len(voltage_range), len(current_range)))
    diff_currents = np.zeros((len(voltage_range), len(current_range)))
    
    # Calculate the current for each combination of voltage and current
    for i, voltage in enumerate(voltage_range):
        for j, current in enumerate(current_range):
            V = voltage
            J = current
            resulting_current = photon_current_density - j_0_pv_cell * np.exp((V + J * r_s) / (CONSTANTS['v_t'] * diode_n)) - (V + J * r_s) / r_sh
            transcendental_current[i, j] = resulting_current
            diff_currents[i, j] = abs(current - resulting_current)
    
    # Find the current where the difference between calculated and actual is minimal
    min_current_diff = np.min(diff_currents, axis=1)
    min_current_diff_mask = (diff_currents == min_current_diff[:, None])
    solved_current = transcendental_current[min_current_diff_mask]
    
    # Calculate power and find its maximum value
    power = solved_current * voltage_range
    max_power = np.max(power)
    
    # Calculate PCE as a percentage
    pce = max_power / 1000 * 100  # %

    return pce


