# -*- coding: utf-8 -*-

# Standard libraries
import os
import math
import warnings

# Third-party libraries
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d as p1d
from scipy.integrate import quad
from scipy.constants import c as speed_of_light, h as planck_constant

# Local module imports
from .import_data_helper_functions import (
    import_spectrum,
    import_quantum_yield
)
from .colorimetry_and_efficiency_functions import (
    calculate_cri, 
    calculate_avt, 
    calculate_pce
)

# To avoid cluttering with pandas warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) 

current_dir = os.getcwd()

def sychugov_numerator(l):
    return 1 / (2 * math.pi * math.sqrt(l**2 - 1))

def sychugov_denominator(l):
    return 1 / (2 * math.pi * l * math.sqrt(l**2 - 1))

def calculate_correction_factor(lower_bound, upper_bound):
    """
    Calculate the correction factor using the Sychugov numerator and denominator.
    
    :param lower_bound: The lower bound for integration
    :param upper_bound: The upper bound for integration
    :return: The correction factor value
    """
    numerator = quad(sychugov_numerator, lower_bound, upper_bound)
    denominator = quad(sychugov_denominator, lower_bound, upper_bound)
    return numerator[0] / denominator[0]

def path_length_probability_distribution(l,  width, height, diagonal, k_correction_factor):
    """
    Calculate the probability distribution as a function of 
    the path length inside the LSC as provided by Sychugov (2019).

    :param l: The path length
    :param width: The width of the LSC
    :param height: The height of the LSC
    :param diagonal: The diagonal of the LSC
    :param k_correction_factor: The correction factor
    :return: The probability distribution value
    """
    k, w, h, d = k_correction_factor, width, height, diagonal
    
    # Define conditions for probability calculations
    conditions = [
        (l > d * k),
        (l >= 0 and l <= k * w),
        (l >= k * w and l <= h * k),
        (l >= h * k and l <= d * k)
    ]
    
    # Define corresponding probability calculations
    calculations = [
        0,
        (2*w + 2*h - 2*l/k) / (np.pi * h * w * k),
        (2*l - 2*np.sqrt(l**2 - (k* w)**2)) / (np.pi * l * w * k),
        (2*(l**2)/k- 2*h*np.sqrt(l**2 - (k* w)**2) - 2*w*np.sqrt(l**2 - (k*h)**2)) / (np.pi * l * h * w * k)
    ]
    
    for condition, calculation in zip(conditions, calculations):
        if condition:
            return calculation



def numerical_single(luminophore, concentration, parameter_dict):     
    
    
    # Set start and end of wavelength distribution range
    spectrum_start = parameter_dict["spectrum_start"]
    spectrum_end = parameter_dict["spectrum_end"]
    interval = parameter_dict["interval"]

     # Import light spectrum, absorption, and emission using the combined function
    light_spectrum = import_spectrum("AM1.5G", spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval, plotting=False)
    absorption_df = import_spectrum("absorption", name=luminophore, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
    emission_df = import_spectrum("emission", name=luminophore, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)

    # Handle quantum yield
    if parameter_dict.get("quantum_yield_unity", False):
        quantum_yield = 1.0
    else:
        quantum_yield = import_quantum_yield(luminophore)
        
    # Set LSC dimensions
    width = parameter_dict["width"]
    height = parameter_dict["height"]
    thickness = parameter_dict["thickness"]    

    if height < width:
        raise ValueError("Width > height: Due to set-up of Sychugov algorithm, height should be larger than width.")

    # Calculate escape cone loss based on isotropic emission
    refractive_index_wg = parameter_dict["refractive_index_wg"]
    refractive_index_air = parameter_dict["refractive_index_air"]

    # From McDowall et al., (2013) Applied Optics Vol. 52, Issue 6, pp. 1230-1239 https://doi.org/10.1364/AO.52.001230
    escape_cone_loss = 1 - math.sqrt(1 - (refractive_index_air/refractive_index_wg)**2) 

    # Import PMMA spectrum
    pmma_coefficient = pd.read_csv(os.path.join(current_dir, "Data", "PMMA_coefficient.csv"), sep=';', decimal='.')

    pmma_coefficient_interpolation = p1d(pmma_coefficient.wavelength, pmma_coefficient['spectrum'])
    new_range = np.arange(spectrum_start, spectrum_end + interval, interval)

    # Option to set the waveguide absorption to very low values.
    factor_pmma = 1E-5 if parameter_dict.get('waveguide_absorption_off', False) else 1.0
        
    pmma_coefficient_values = pmma_coefficient_interpolation(new_range) * factor_pmma
    pmma_coefficient = pd.DataFrame({'wavelength': new_range, 'spectrum': pmma_coefficient_values})

    # Calculate path distribution for waveguide
    path_length_interval = 0.1

    k_correction_factor = calculate_correction_factor(refractive_index_air, refractive_index_wg)

    diagonal = np.sqrt(width**2 + height**2)
    max_path_length = diagonal * k_correction_factor

    # Length of array (make sure it's square)
    array_length = (spectrum_end - spectrum_start) / interval
    if array_length % 1 != 0:
        raise ValueError("Array has to be of a length which floors to zero.")

    photon_path_lengths = np.arange(0, max_path_length, path_length_interval)
    path_length_probabilities = [path_length_probability_distribution(r, width, height, diagonal, k_correction_factor) for r in photon_path_lengths]
    total_probability = sum(path_length_probabilities)

    normalized_probabilities = [p / total_probability for p in path_length_probabilities]

    # Ensure normalized_probabilities has two dimensions.
    normalized_probabilities = np.array(normalized_probabilities)[:, np.newaxis]


    # === INITIAL ABSORPTION CALCULATIONS ===

    # Calculate the fresnel transmission. This accounts for the reflection loss at the interface of two different mediums.
    fresnel_transmission = 1 - ((refractive_index_air - refractive_index_wg) / (refractive_index_air + refractive_index_wg))**2

    # Convert pandas series to numpy arrays for faster calculation.
    wavelengths = light_spectrum.wavelength.values
    spectrum = light_spectrum.spectrum.values
    absorption_values = absorption_df.absorption.values

    # Calculate the spectrum after reflection loss.
    spectrum_after_reflection = spectrum * fresnel_transmission

    # Calculate the absorption fraction combining the effects of the luminophore and waveguide.
    combined_absorption = absorption_values * concentration + pmma_coefficient.spectrum.values
    fraction_luminophore_and_wg = 1 - np.exp(-thickness * combined_absorption)

    # Compute the absorbed photons by both luminophore and waveguide.
    absorption_luminophore_and_wg = fraction_luminophore_and_wg * spectrum_after_reflection

    # Determine the ratio of absorbed photons due to the luminophore.
    abs_ratio_lum = absorption_values * concentration / combined_absorption

    # Calculate total absorbed photons by the luminophore.
    total_absorbed_photons_init = np.sum(absorption_luminophore_and_wg * abs_ratio_lum)

    # === CALCULATE ABSORPTION PROBABILITIES ===

    # Calculate the probability of a photon being absorbed outside the escape cone (EC).
    absorption_outside_EC = np.outer(absorption_df.absorption * concentration + pmma_coefficient.spectrum,  photon_path_lengths)
    probabilities_outside_EC = np.exp(-absorption_outside_EC) * normalized_probabilities.T

    # Create a dataframe with probabilities of a photon reaching the edge.
    probabilities = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC.sum(axis=1),
                                  'wavelength': light_spectrum.wavelength})

    # Calculate absorption probabilities within the escape cone.
    critical_angle = math.asin(refractive_index_air/refractive_index_wg)
    k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
    half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption))
    depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption

    probabilities['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption)
    probabilities['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption)

    # Normalize emission spectrum
    probabilities['normalized_emission'] = emission_df.emission / emission_df.emission.sum()


    # === CALCULATE EMISSION EVENTS ===

    # Set the number of re-emissions to be considered.
    number_of_re_emissions = 5

    # Initialize variables to store results
    wavelengths = light_spectrum.wavelength.values
    emitted_photons = [total_absorbed_photons_init * quantum_yield]
    photons_reaching_the_edge = []

    # Calculate values that are used repeatedly outside the loop
    norm_emit_escape = probabilities.normalized_emission.values * escape_cone_loss / 2
    norm_emit_outside = probabilities.normalized_emission.values * (1 - escape_cone_loss)
    prob_photon_edge = probabilities.probability_photon_reaching_edge.values
    abs_prob_in_ec_up = probabilities.absorption_probs_in_ec_up.values
    abs_prob_in_ec_down = probabilities.absorption_probs_in_ec_down.values

    # Lists to store parameters
    emitted_photons_list = [emitted_photons[0]]  # Start with initial emitted photons
    absorbed_photons_list = []
    photons_emitted_through_bottom_list = []


    # Loop for re-emissions
    for i in range(number_of_re_emissions):
        emitted_photons_current = emitted_photons_list[-1]  # Get the last value
        
        # Calculations for emitted photons
        emitted_up_in_ec = emitted_photons_current * norm_emit_escape
        emitted_down_in_ec = emitted_photons_current * norm_emit_escape
        emitted_out_ec = emitted_photons_current * norm_emit_outside
        
        # Photons reaching the edge
        reaching_edge = prob_photon_edge * emitted_out_ec
        photons_reaching_the_edge.append(reaching_edge)

        # Calculate absorbed photons
        absorbed_up_in_ec = abs_prob_in_ec_up * emitted_up_in_ec  * abs_ratio_lum
        absorbed_down_in_ec = abs_prob_in_ec_down * emitted_down_in_ec  * abs_ratio_lum
        absorbed_out_ec = (1 - prob_photon_edge) * emitted_out_ec * abs_ratio_lum

        # Calculate photons transmitted through bottom 
        photons_emitted_through_bottom_list.append((1 - abs_prob_in_ec_down) * emitted_down_in_ec)

        # Total absorbed photons for luminophore
        total_absorbed = absorbed_up_in_ec + absorbed_down_in_ec + absorbed_out_ec
        
        # Update the next round of emissions
        emitted_photons_list.append(total_absorbed.sum() * quantum_yield)
        absorbed_photons_list.append(total_absorbed)


    # Calculate transmission fraction for the luminophore and waveguide combined
    transmission_fraction_luminophore_and_wg = np.exp(-thickness * (absorption_df.absorption.values * concentration + pmma_coefficient.spectrum.values))
    fresnel_transmission_fraction = fresnel_transmission**2
    initial_transmission = light_spectrum.spectrum.values * transmission_fraction_luminophore_and_wg * fresnel_transmission_fraction
    transmission_from_emission = np.sum(photons_emitted_through_bottom_list, axis=0)
    transmission_spectrum = initial_transmission + transmission_from_emission

    # AVT Calculation
    avt = calculate_avt(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)

    # CRI Calculation
    transmission_spectrum = (speed_of_light * planck_constant * transmission_spectrum) / (light_spectrum.wavelength * 1e-9) / interval
      
    try:
       cri = calculate_cri(wavelengths_of_spectrum = wavelengths, transmitted_spectrum = transmission_spectrum, interval = interval)
    except ZeroDivisionError:
       print("ZeroDivisionerror in CRI calculation, setting at 0.0")
       cri = 0.0
          
    side_incident_photons_total = np.sum(photons_reaching_the_edge, axis=0)
        
    pce = calculate_pce(wavelengths,
                        side_incident_photons_total,
                        spectrum_start,
                        spectrum_end,
                        interval)

    
    return pce, avt, cri

def numerical_double(luminophore_1, concentration_1, luminophore_2, concentration_2, parameter_dict):

   # Set start and end of wavelength distribution range (works for light, but also luminophore abs and ems limits)
   spectrum_start = parameter_dict["spectrum_start"]
   spectrum_end = parameter_dict["spectrum_end"]
   interval = parameter_dict["interval"] 

   # Import light spectrum, absorption, and emission using the combined function
   light_spectrum = import_spectrum("AM1.5G", spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval, plotting=False)

   absorption_df_1 = import_spectrum("absorption", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
   emission_df_1 = import_spectrum("emission", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)

   # Handle quantum yield
   if parameter_dict.get("quantum_yield_unity", False):
       quantum_yield_1 = 1.0
   else:
       quantum_yield_1 = import_quantum_yield(luminophore_1)

   # Import light spectrum, absorption, and emission using the combined function
   absorption_df_2 = import_spectrum("absorption", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
   emission_df_2 = import_spectrum("emission", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
      
   # Handle quantum yield
   if parameter_dict.get("quantum_yield_unity", False):
       quantum_yield_2 = 1.0
   else:
       quantum_yield_2 = import_quantum_yield(luminophore_2)
       

   # Set LSC dimensions
   width = parameter_dict["width"]
   height = parameter_dict["height"]
   thickness = parameter_dict["thickness"] 
       
   # Calculate escape cone loss based on isotropic emission
   refractive_index_wg = parameter_dict["refractive_index_wg"]
   refractive_index_air = parameter_dict["refractive_index_air"]

   # From McDowall et al., (2013) Applied Optics Vol. 52, Issue 6, pp. 1230-1239 https://doi.org/10.1364/AO.52.001230
   escape_cone_loss = 1 - math.sqrt(1 - (refractive_index_air/refractive_index_wg)**2) 

   # Import PMMA spectrum
   pmma_coefficient = pd.read_csv(os.path.join(current_dir, "Data", "PMMA_coefficient.csv"), sep=';', decimal='.')

   pmma_coefficient_interpolation = p1d(pmma_coefficient.wavelength, pmma_coefficient['spectrum'])
   new_range = np.arange(spectrum_start, spectrum_end + interval, interval)

   # Option to set the waveguide absorption to very low values.
   factor_pmma = 1E-5 if parameter_dict.get('waveguide_absorption_off', False) else 1.0
       
   pmma_coefficient_values = pmma_coefficient_interpolation(new_range) * factor_pmma
   pmma_coefficient = pd.DataFrame({'wavelength': new_range, 'spectrum': pmma_coefficient_values})

   # Calculate path distribution for waveguide
   path_length_interval = 0.1

   k_correction_factor = calculate_correction_factor(refractive_index_air, refractive_index_wg)

   diagonal = np.sqrt(width**2 + height**2)
   max_path_length = diagonal * k_correction_factor

   # Length of array (make sure it's square)
   array_length = (spectrum_end - spectrum_start) / interval
   if array_length % 1 != 0:
       raise ValueError("Array has to be of a length which floors to zero.")

   photon_path_lengths = np.arange(0, max_path_length, path_length_interval)
   path_length_probabilities = [path_length_probability_distribution(r, width, height, diagonal, k_correction_factor) for r in photon_path_lengths]
   total_probability = sum(path_length_probabilities)

   normalized_probabilities = [p / total_probability for p in path_length_probabilities]

   # Ensure normalized_probabilities has two dimensions.
   normalized_probabilities = np.array(normalized_probabilities)[:, np.newaxis]
      

   # === INITIAL ABSORPTION CALCULATIONS FOR TANDEM LSC ===

   # Calculate the fresnel transmission accounting for the reflection loss at the interface of two different mediums.
   fresnel_transmission = 1 - ((refractive_index_air - refractive_index_wg) / (refractive_index_air + refractive_index_wg))**2

   # Convert pandas series to numpy arrays for faster calculation.
   wavelengths = light_spectrum.wavelength.values
   spectrum = light_spectrum.spectrum.values
   absorption_values_1 = absorption_df_1.absorption.values
   absorption_values_2 = absorption_df_2.absorption.values

   # Calculate the spectrum after reflection loss.
   spectrum_after_reflection = spectrum * fresnel_transmission

   # Calculate combined absorption and absorption fraction for the first luminophore and waveguide.
   combined_absorption_1 = absorption_values_1 * concentration_1 + pmma_coefficient.spectrum.values
   fraction_luminophore_and_wg_1 = 1 - np.exp(-thickness * combined_absorption_1)
   absorption_luminophore_and_wg_1 = fraction_luminophore_and_wg_1 * spectrum_after_reflection
   abs_ratio_lum_1 = absorption_values_1 * concentration_1 / combined_absorption_1
   transmission_after_lsc_1 = (spectrum_after_reflection - absorption_luminophore_and_wg_1) * fresnel_transmission**2
   total_absorbed_photons_init_1 = np.sum(absorption_luminophore_and_wg_1 * abs_ratio_lum_1)


   # Calculate combined absorption and absorption fraction for the second luminophore and waveguide.
   combined_absorption_2 = absorption_values_2 * concentration_2 + pmma_coefficient.spectrum.values
   fraction_luminophore_and_wg_2 = 1 - np.exp(-thickness * combined_absorption_2)
   absorption_luminophore_and_wg_2 = fraction_luminophore_and_wg_2 * transmission_after_lsc_1
   abs_ratio_lum_2 = absorption_values_2 * concentration_2 / combined_absorption_2
   initial_transmission = (transmission_after_lsc_1 - absorption_luminophore_and_wg_2) * fresnel_transmission
   total_absorbed_photons_init_2 = np.sum(absorption_luminophore_and_wg_2 * abs_ratio_lum_2)


   # === CALCULATE ABSORPTION PROBABILITIES LSC 1 ===

   # Calculate the probability of a photon being absorbed outside the escape cone (EC).
   absorption_outside_EC_1 = np.outer(combined_absorption_1,  photon_path_lengths)
   probabilities_outside_EC_1 = np.exp(-absorption_outside_EC_1) * normalized_probabilities.T

   # Create a dataframe with probabilities of a photon reaching the edge.
   probabilities_1 = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC_1.sum(axis=1),
                                   'wavelength': light_spectrum.wavelength})

   # Calculate absorption probabilities within the escape cone.
   critical_angle = math.asin(refractive_index_air/refractive_index_wg)
   k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
   half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption_1))
   depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption_1

   probabilities_1['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption_1)
   probabilities_1['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption_1)

   # Normalize emission spectrum
   probabilities_1['normalized_emission'] = emission_df_1.emission / emission_df_1.emission.sum()

   # === CALCULATE ABSORPTION PROBABILITIES LSC 2 ===

   # Calculate the probability of a photon being absorbed outside the escape cone (EC).
   absorption_outside_EC_2 = np.outer(combined_absorption_2,  photon_path_lengths)
   probabilities_outside_EC_2 = np.exp(-absorption_outside_EC_2) * normalized_probabilities.T

   # Create a dataframe with probabilities of a photon reaching the edge.
   probabilities_2 = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC_2.sum(axis=1),
                                   'wavelength': light_spectrum.wavelength})

   # Calculate absorption probabilities within the escape cone.
   critical_angle = math.asin(refractive_index_air/refractive_index_wg)
   k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
   half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption_2))
   depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption_2

   probabilities_2['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption_2)
   probabilities_2['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption_2)

   # Normalize emission spectrum
   probabilities_2['normalized_emission'] = emission_df_2.emission / emission_df_2.emission.sum()



   # === CALCULATE EMISSION EVENTS ===
    
   # Set the number of re-emissions to be considered.
   number_of_re_emissions = 5
    
   # Initialize variables to store results
   emitted_photons_list_1 = [total_absorbed_photons_init_1 * quantum_yield_1]
   emitted_photons_list_2 = [total_absorbed_photons_init_2 * quantum_yield_2]
   photons_reaching_the_edge_1 = []
   photons_reaching_the_edge_2 = []   
   photons_transmitted_through_bottom_after_ems_by_lum1 = []
   photons_transmitted_through_bottom_after_ems_by_lum2 = []


   # Calculate values that are used repeatedly outside the loop for LSC 1
   norm_emit_escape_1 = probabilities_1.normalized_emission.values * escape_cone_loss / 2
   norm_emit_outside_1 = probabilities_1.normalized_emission.values * (1 - escape_cone_loss)
   prob_photon_edge_1 = probabilities_1.probability_photon_reaching_edge.values
   abs_prob_in_ec_up_1 = probabilities_1.absorption_probs_in_ec_up.values
   abs_prob_in_ec_down_1 = probabilities_1.absorption_probs_in_ec_down.values
    
   # Calculate values that are used repeatedly outside the loop for LSC 2
   norm_emit_escape_2 = probabilities_2.normalized_emission.values * escape_cone_loss / 2
   norm_emit_outside_2 = probabilities_2.normalized_emission.values * (1 - escape_cone_loss)
   prob_photon_edge_2 = probabilities_2.probability_photon_reaching_edge.values
   abs_prob_in_ec_up_2 = probabilities_2.absorption_probs_in_ec_up.values
   abs_prob_in_ec_down_2 = probabilities_2.absorption_probs_in_ec_down.values
       
     
   for i in range(number_of_re_emissions):
       
       emitted_photons_current_1 = emitted_photons_list_1[-1]  # Get the last value
       emitted_photons_current_2 = emitted_photons_list_2[-1]  # Get the last value

       # Calculations for emitted photons in LSC 1
       emitted_up_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
       emitted_down_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
       emitted_out_ec_1 = emitted_photons_current_1 * norm_emit_outside_1
       
       # Photons reaching the edge LSC 1
       reaching_edge_1 = prob_photon_edge_1 * emitted_out_ec_1
       photons_reaching_the_edge_1.append(reaching_edge_1)
       
       # Calculations for emitted photons in LSC 2
       emitted_up_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
       emitted_down_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
       emitted_out_ec_2 = emitted_photons_current_2 * norm_emit_outside_2
       
       # Photons reaching the edge LSC 2
       reaching_edge_2 = prob_photon_edge_2 * emitted_out_ec_2
       photons_reaching_the_edge_2.append(reaching_edge_2)

       # Calculate absorbed photons LSC 1
       absorbed_up_in_ec_1 = abs_prob_in_ec_up_1 * emitted_up_in_ec_1  * abs_ratio_lum_1
       absorbed_down_in_ec_1 = abs_prob_in_ec_down_1 * emitted_down_in_ec_1  * abs_ratio_lum_1
       absorbed_out_ec_1 = (1 - prob_photon_edge_1) * emitted_out_ec_1 * abs_ratio_lum_1

       # Calculate absorption after emission by LSC 1 or 2 and in the other LSC
       photons_transmitted_towards_lum_1 =  emitted_up_in_ec_2 * ( 1- abs_prob_in_ec_up_2)
       absorbed_photons_by_lum_1_from_emission_lum_2 = fraction_luminophore_and_wg_1 * photons_transmitted_towards_lum_1 * abs_ratio_lum_1
       photons_transmitted_towards_lum_2  = (1 - abs_prob_in_ec_down_1) * emitted_down_in_ec_1
       absorbed_photons_by_lum_2_from_emission_lum_1 = fraction_luminophore_and_wg_2 * photons_transmitted_towards_lum_2 * abs_ratio_lum_2
       
       # Calculate absorbed photons LSC 2
       absorbed_up_in_ec_2 = abs_prob_in_ec_up_2 * emitted_up_in_ec_2  * abs_ratio_lum_2
       absorbed_down_in_ec_2 = abs_prob_in_ec_down_2 * emitted_down_in_ec_2  * abs_ratio_lum_2
       absorbed_out_ec_2 = (1 - prob_photon_edge_2) * emitted_out_ec_2 * abs_ratio_lum_2

       # Photons emitted through bottom after emission by lum 1 and lum 2
       photons_transmitted_through_bottom_after_ems_by_lum1.append((1 -  fraction_luminophore_and_wg_2) * photons_transmitted_towards_lum_2)
       photons_transmitted_through_bottom_after_ems_by_lum2.append((1 - abs_prob_in_ec_down_2) * emitted_down_in_ec_2)

       # Total absorbed photons for luminophore
       total_absorbed_1 = absorbed_up_in_ec_1 + absorbed_down_in_ec_1 + absorbed_out_ec_1 + absorbed_photons_by_lum_1_from_emission_lum_2
       total_absorbed_2 = absorbed_up_in_ec_2 + absorbed_down_in_ec_2 + absorbed_out_ec_2 + absorbed_photons_by_lum_2_from_emission_lum_1
       
       # Update the next round of emissions
       emitted_photons_list_1.append(total_absorbed_1.sum() * quantum_yield_1)
       emitted_photons_list_2.append(total_absorbed_2.sum() * quantum_yield_2)


   # Calculate transmission fraction for the luminophore and waveguide combined
   transmission_from_emission = np.sum(photons_transmitted_through_bottom_after_ems_by_lum1, axis=0) + np.sum(photons_transmitted_through_bottom_after_ems_by_lum2, axis=0)
   transmission_spectrum = initial_transmission + transmission_from_emission
   
   # AVT Calculation
   avt = calculate_avt(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)
    
   # CRI Calculation
   transmission_spectrum = (speed_of_light * planck_constant * transmission_spectrum) / (light_spectrum.wavelength * 1e-9) / interval
    
   try:
        cri = calculate_cri(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)
   except ZeroDivisionError:
        print("ZeroDivisionerror in CRI calculation, setting at 0.0")
        cri = 0.0
          
         
   side_incident_photons_total = np.sum(photons_reaching_the_edge_1, axis=0) + np.sum(photons_reaching_the_edge_2, axis=0)

   pce = calculate_pce(wavelengths,
                        side_incident_photons_total,
                        spectrum_start,
                        spectrum_end,
                        interval)     
        
   return pce, avt, cri

    
def numerical_triple(luminophore_1, concentration_1, luminophore_2, concentration_2, luminophore_3, concentration_3, parameter_dict):
    
    
    # Set start and end of wavelength distribution range (works for light, but also luminophore abs and ems limits)
    spectrum_start = parameter_dict["spectrum_start"]
    spectrum_end = parameter_dict["spectrum_end"]
    interval = parameter_dict["interval"] 
    
    # Import light spectrum, absorption, and emission using the combined function
    light_spectrum = import_spectrum("AM1.5G", spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval, plotting=False)
    
    absorption_df_1 = import_spectrum("absorption", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
    emission_df_1 = import_spectrum("emission", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
    
    # Handle quantum yield
    if parameter_dict.get("quantum_yield_unity", False):
        quantum_yield_1 = 1.0
    else:
        quantum_yield_1 = import_quantum_yield(luminophore_1)
    
    # Import light spectrum, absorption, and emission using the combined function
    absorption_df_2 = import_spectrum("absorption", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
    emission_df_2 = import_spectrum("emission", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
       
    # Handle quantum yield
    if parameter_dict.get("quantum_yield_unity", False):
        quantum_yield_2 = 1.0
    else:
        quantum_yield_2 = import_quantum_yield(luminophore_2)
        
    # Import light spectrum, absorption, and emission using the combined function
    absorption_df_3 = import_spectrum("absorption", name=luminophore_3, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
    emission_df_3 = import_spectrum("emission", name=luminophore_3, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
        
    # Handle quantum yield
    if parameter_dict.get("quantum_yield_unity", False):
         quantum_yield_3 = 1.0
    else:
         quantum_yield_3 = import_quantum_yield(luminophore_3)
        
    
    # Set LSC dimensions
    width = parameter_dict["width"]
    height = parameter_dict["height"]
    thickness = parameter_dict["thickness"] 
        
    # Calculate escape cone loss based on isotropic emission
    refractive_index_wg = parameter_dict["refractive_index_wg"]
    refractive_index_air = parameter_dict["refractive_index_air"]
    
    # From McDowall et al., (2013) Applied Optics Vol. 52, Issue 6, pp. 1230-1239 https://doi.org/10.1364/AO.52.001230
    escape_cone_loss = 1 - math.sqrt(1 - (refractive_index_air/refractive_index_wg)**2) 
    
    # Import PMMA spectrum
    pmma_coefficient = pd.read_csv(os.path.join(current_dir, "Data", "PMMA_coefficient.csv"), sep=';', decimal='.')
    
    pmma_coefficient_interpolation = p1d(pmma_coefficient.wavelength, pmma_coefficient['spectrum'])
    new_range = np.arange(spectrum_start, spectrum_end + interval, interval)
    
    # Option to set the waveguide absorption to very low values.
    factor_pmma = 1E-5 if parameter_dict.get('waveguide_absorption_off', False) else 1.0
        
    pmma_coefficient_values = pmma_coefficient_interpolation(new_range) * factor_pmma
    pmma_coefficient = pd.DataFrame({'wavelength': new_range, 'spectrum': pmma_coefficient_values})
    
    # Calculate path distribution for waveguide
    path_length_interval = 0.1
    
    k_correction_factor = calculate_correction_factor(refractive_index_air, refractive_index_wg)
    
    diagonal = np.sqrt(width**2 + height**2)
    max_path_length = diagonal * k_correction_factor
    
    # Length of array (make sure it's square)
    array_length = (spectrum_end - spectrum_start) / interval
    if array_length % 1 != 0:
        raise ValueError("Array has to be of a length which floors to zero.")
    
    photon_path_lengths = np.arange(0, max_path_length, path_length_interval)
    path_length_probabilities = [path_length_probability_distribution(r, width, height, diagonal, k_correction_factor) for r in photon_path_lengths]
    total_probability = sum(path_length_probabilities)
    
    normalized_probabilities = [p / total_probability for p in path_length_probabilities]
    
    # Ensure normalized_probabilities has two dimensions.
    normalized_probabilities = np.array(normalized_probabilities)[:, np.newaxis]
       
    
    # === INITIAL ABSORPTION CALCULATIONS FOR TANDEM LSC ===
     
    # Calculate the fresnel transmission accounting for the reflection loss at the interface of two different mediums.
    fresnel_transmission = 1 - ((refractive_index_air - refractive_index_wg) / (refractive_index_air + refractive_index_wg))**2
     
    # Convert pandas series to numpy arrays for faster calculation.
    wavelengths = light_spectrum.wavelength.values
    spectrum = light_spectrum.spectrum.values
    absorption_values_1 = absorption_df_1.absorption.values
    absorption_values_2 = absorption_df_2.absorption.values
    absorption_values_3 = absorption_df_3.absorption.values

     
    # Calculate the spectrum after reflection loss.
    spectrum_after_reflection = spectrum * fresnel_transmission
     
    # Calculate combined absorption and absorption fraction for the first luminophore and waveguide.
    combined_absorption_1 = absorption_values_1 * concentration_1 + pmma_coefficient.spectrum.values
    fraction_luminophore_and_wg_1 = 1 - np.exp(-thickness * combined_absorption_1)
    absorption_luminophore_and_wg_1 = fraction_luminophore_and_wg_1 * spectrum_after_reflection
    abs_ratio_lum_1 = absorption_values_1 * concentration_1 / combined_absorption_1
    transmission_after_lsc_1 = (spectrum_after_reflection - absorption_luminophore_and_wg_1) * fresnel_transmission**2
    total_absorbed_photons_init_1 = np.sum(absorption_luminophore_and_wg_1 * abs_ratio_lum_1)
     
     
    # Calculate combined absorption and absorption fraction for the second luminophore and waveguide.
    combined_absorption_2 = absorption_values_2 * concentration_2 + pmma_coefficient.spectrum.values
    fraction_luminophore_and_wg_2 = 1 - np.exp(-thickness * combined_absorption_2)
    absorption_luminophore_and_wg_2 = fraction_luminophore_and_wg_2 * transmission_after_lsc_1
    abs_ratio_lum_2 = absorption_values_2 * concentration_2 / combined_absorption_2
    transmission_after_lsc_2 = (transmission_after_lsc_1 - absorption_luminophore_and_wg_2) * fresnel_transmission**2
    total_absorbed_photons_init_2 = np.sum(absorption_luminophore_and_wg_2 * abs_ratio_lum_2)
    
    # Calculate combined absorption and absorption fraction for the third luminophore and waveguide.
    combined_absorption_3 = absorption_values_3 * concentration_3 + pmma_coefficient.spectrum.values
    fraction_luminophore_and_wg_3 = 1 - np.exp(-thickness * combined_absorption_3)
    absorption_luminophore_and_wg_3 = fraction_luminophore_and_wg_3 * transmission_after_lsc_2
    abs_ratio_lum_3 = absorption_values_3 * concentration_3 / combined_absorption_3
    initial_transmission = (transmission_after_lsc_2 - absorption_luminophore_and_wg_3) * fresnel_transmission
    total_absorbed_photons_init_3 = np.sum(absorption_luminophore_and_wg_3 * abs_ratio_lum_3)

    # === CALCULATE ABSORPTION PROBABILITIES LSC 1 ===
    
    # Calculate the probability of a photon being absorbed outside the escape cone (EC).
    absorption_outside_EC_1 = np.outer(combined_absorption_1,  photon_path_lengths)
    probabilities_outside_EC_1 = np.exp(-absorption_outside_EC_1) * normalized_probabilities.T
    
    # Create a dataframe with probabilities of a photon reaching the edge.
    probabilities_1 = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC_1.sum(axis=1),
                                    'wavelength': light_spectrum.wavelength})
    
    # Calculate absorption probabilities within the escape cone.
    critical_angle = math.asin(refractive_index_air/refractive_index_wg)
    k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
    half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption_1))
    depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption_1
    
    probabilities_1['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption_1)
    probabilities_1['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption_1)
    
    # Normalize emission spectrum
    probabilities_1['normalized_emission'] = emission_df_1.emission / emission_df_1.emission.sum()
    
    # === CALCULATE ABSORPTION PROBABILITIES LSC 2 ===
    
    # Calculate the probability of a photon being absorbed outside the escape cone (EC).
    absorption_outside_EC_2 = np.outer(combined_absorption_2,  photon_path_lengths)
    probabilities_outside_EC_2 = np.exp(-absorption_outside_EC_2) * normalized_probabilities.T
    
    # Create a dataframe with probabilities of a photon reaching the edge.
    probabilities_2 = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC_2.sum(axis=1),
                                    'wavelength': light_spectrum.wavelength})
    
    # Calculate absorption probabilities within the escape cone.
    critical_angle = math.asin(refractive_index_air/refractive_index_wg)
    k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
    half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption_2))
    depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption_2
    
    probabilities_2['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption_2)
    probabilities_2['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption_2)
    
    # Normalize emission spectrum
    probabilities_2['normalized_emission'] = emission_df_2.emission / emission_df_2.emission.sum()
    
    # === CALCULATE ABSORPTION PROBABILITIES LSC 3 ===
    
    # Calculate the probability of a photon being absorbed outside the escape cone (EC).
    absorption_outside_EC_3 = np.outer(combined_absorption_3,  photon_path_lengths)
    probabilities_outside_EC_3 = np.exp(-absorption_outside_EC_3) * normalized_probabilities.T
    
    # Create a dataframe with probabilities of a photon reaching the edge.
    probabilities_3 = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC_3.sum(axis=1),
                                    'wavelength': light_spectrum.wavelength})
    
    # Calculate absorption probabilities within the escape cone.
    critical_angle = math.asin(refractive_index_air/refractive_index_wg)
    k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
    half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption_3))
    depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption_3
    
    probabilities_3['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption_3)
    probabilities_3['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption_3)
    
    # Normalize emission spectrum
    probabilities_3['normalized_emission'] = emission_df_3.emission / emission_df_3.emission.sum()
    
    
    # === CALCULATE EMISSION EVENTS ===
     
    # Set the number of re-emissions to be considered.
    number_of_re_emissions = 5
     
    # Initialize variables to store results
    emitted_photons_list_1 = [total_absorbed_photons_init_1 * quantum_yield_1]
    emitted_photons_list_2 = [total_absorbed_photons_init_2 * quantum_yield_2]
    emitted_photons_list_3 = [total_absorbed_photons_init_3 * quantum_yield_3]
    photons_reaching_the_edge_1 = []
    photons_reaching_the_edge_2 = []   
    photons_reaching_the_edge_3 = []   
    photons_transmitted_through_bottom_after_ems_by_lum1_and_2 = []
    photons_transmitted_through_bottom_after_ems_by_lum3 = []
    
    
    # Calculate values that are used repeatedly outside the loop for LSC 1
    norm_emit_escape_1 = probabilities_1.normalized_emission.values * escape_cone_loss / 2
    norm_emit_outside_1 = probabilities_1.normalized_emission.values * (1 - escape_cone_loss)
    prob_photon_edge_1 = probabilities_1.probability_photon_reaching_edge.values
    abs_prob_in_ec_up_1 = probabilities_1.absorption_probs_in_ec_up.values
    abs_prob_in_ec_down_1 = probabilities_1.absorption_probs_in_ec_down.values
     
    # Calculate values that are used repeatedly outside the loop for LSC 2
    norm_emit_escape_2 = probabilities_2.normalized_emission.values * escape_cone_loss / 2
    norm_emit_outside_2 = probabilities_2.normalized_emission.values * (1 - escape_cone_loss)
    prob_photon_edge_2 = probabilities_2.probability_photon_reaching_edge.values
    abs_prob_in_ec_up_2 = probabilities_2.absorption_probs_in_ec_up.values
    abs_prob_in_ec_down_2 = probabilities_2.absorption_probs_in_ec_down.values
    
    # Calculate values that are used repeatedly outside the loop for LSC 3
    norm_emit_escape_3 = probabilities_3.normalized_emission.values * escape_cone_loss / 2
    norm_emit_outside_3 = probabilities_3.normalized_emission.values * (1 - escape_cone_loss)
    prob_photon_edge_3 = probabilities_3.probability_photon_reaching_edge.values
    abs_prob_in_ec_up_3 = probabilities_3.absorption_probs_in_ec_up.values
    abs_prob_in_ec_down_3 = probabilities_3.absorption_probs_in_ec_down.values
     
     
    for i in range(number_of_re_emissions):
         
        emitted_photons_current_1 = emitted_photons_list_1[-1]  # Get the last value
        emitted_photons_current_2 = emitted_photons_list_2[-1]  # Get the last value
        emitted_photons_current_3 = emitted_photons_list_3[-1]  # Get the last value
    
    
        # Calculations for emitted photons in LSC 1
        emitted_up_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
        emitted_down_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
        emitted_out_ec_1 = emitted_photons_current_1 * norm_emit_outside_1
        
        # Photons reaching the edge LSC 1
        reaching_edge_1 = prob_photon_edge_1 * emitted_out_ec_1
        photons_reaching_the_edge_1.append(reaching_edge_1)
        
        # Calculations for emitted photons in LSC 2
        emitted_up_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
        emitted_down_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
        emitted_out_ec_2 = emitted_photons_current_2 * norm_emit_outside_2
        
        # Photons reaching the edge LSC 2
        reaching_edge_2 = prob_photon_edge_2 * emitted_out_ec_2
        photons_reaching_the_edge_2.append(reaching_edge_2)
        
        # Calculations for emitted photons in LSC 3
        emitted_up_in_ec_3 = emitted_photons_current_3 * norm_emit_escape_3
        emitted_down_in_ec_3 = emitted_photons_current_3 * norm_emit_escape_3
        emitted_out_ec_3 = emitted_photons_current_3 * norm_emit_outside_3
        
        # Photons reaching the edge LSC 3
        reaching_edge_3 = prob_photon_edge_3 * emitted_out_ec_3
        photons_reaching_the_edge_3.append(reaching_edge_3)
    
        # Calculate absorbed photons LSC 1
        absorbed_up_in_ec_1 = abs_prob_in_ec_up_1 * emitted_up_in_ec_1  * abs_ratio_lum_1
        absorbed_down_in_ec_1 = abs_prob_in_ec_down_1 * emitted_down_in_ec_1  * abs_ratio_lum_1
        absorbed_out_ec_1 = (1 - prob_photon_edge_1) * emitted_out_ec_1 * abs_ratio_lum_1
    
        # Calculate absorption after emission by LSC 1 or 2 and in the other LSC
        photons_transmitted_towards_lum_1_by_lum_2 =  emitted_up_in_ec_2 * ( 1- abs_prob_in_ec_up_2)
        absorbed_photons_by_lum_1_from_emission_lum_2 = fraction_luminophore_and_wg_1 * photons_transmitted_towards_lum_1_by_lum_2 * abs_ratio_lum_1
        photons_transmitted_towards_lum_2_by_lum_1  = (1 - abs_prob_in_ec_down_1) * emitted_down_in_ec_1
        absorbed_photons_by_lum_2_from_emission_lum_1 = fraction_luminophore_and_wg_2 * photons_transmitted_towards_lum_2_by_lum_1 * abs_ratio_lum_2
        
        # Calculate absorbed photons LSC 2
        absorbed_up_in_ec_2 = abs_prob_in_ec_up_2 * emitted_up_in_ec_2  * abs_ratio_lum_2
        absorbed_down_in_ec_2 = abs_prob_in_ec_down_2 * emitted_down_in_ec_2  * abs_ratio_lum_2
        absorbed_out_ec_2 = (1 - prob_photon_edge_2) * emitted_out_ec_2 * abs_ratio_lum_2
        
        # Calculate absorbed photons LSC 3
        absorbed_up_in_ec_3 = abs_prob_in_ec_up_3 * emitted_up_in_ec_3  * abs_ratio_lum_3
        absorbed_down_in_ec_3 = abs_prob_in_ec_down_3 * emitted_down_in_ec_3  * abs_ratio_lum_3
        absorbed_out_ec_3 = (1 - prob_photon_edge_3) * emitted_out_ec_3 * abs_ratio_lum_3
    
        # Calculate absorption in LSC 3 after emission by lum 1 and lum 2
        photons_transmitted_towards_lum3_after_ems_by_lum1 = photons_transmitted_towards_lum_2_by_lum_1 * (1 - fraction_luminophore_and_wg_2)
        photons_transmitted_towards_lum3_after_ems_by_lum2 = emitted_down_in_ec_2 * (1 - abs_prob_in_ec_down_2)
        absorbed_photons_by_lum3_from_emission_by_lum1_and_2 = fraction_luminophore_and_wg_3 * (photons_transmitted_towards_lum3_after_ems_by_lum1 + photons_transmitted_towards_lum3_after_ems_by_lum2) * abs_ratio_lum_3
    
        # Calculate photons re-absorbed by 2 after emission by 3
        photons_transmitted_towards_lum2_from_lum3 = (1 - abs_prob_in_ec_up_3) * emitted_up_in_ec_3
        absorbed_photons_from_emission_by_lum3_in_lum2 = fraction_luminophore_and_wg_2 * photons_transmitted_towards_lum2_from_lum3  * abs_ratio_lum_2
        
        # Calculate re-absorption between LSC 1 and 3
        photons_transmitted_through_lum2_from_lum3 = (1 - fraction_luminophore_and_wg_2) * photons_transmitted_towards_lum2_from_lum3 
        absorbed_photons_from_emission_by_lum3_in_lum1 = fraction_luminophore_and_wg_1 * photons_transmitted_through_lum2_from_lum3  * abs_ratio_lum_1
    
        # Photons emitted through bottom after emission by lum 1, lum 2, and lum 3
        photons_transmitted_through_bottom_after_ems_by_lum1_and_2.append((1 -  fraction_luminophore_and_wg_3) * (photons_transmitted_towards_lum3_after_ems_by_lum1 + photons_transmitted_towards_lum3_after_ems_by_lum2))
        photons_transmitted_through_bottom_after_ems_by_lum3.append((1 - abs_prob_in_ec_down_3) * emitted_down_in_ec_3)
    
        # Total absorbed photons for luminophore
        total_absorbed_1 = absorbed_up_in_ec_1 + absorbed_down_in_ec_1 + absorbed_out_ec_1 + absorbed_photons_by_lum_1_from_emission_lum_2 + absorbed_photons_from_emission_by_lum3_in_lum1 
        total_absorbed_2 = absorbed_up_in_ec_2 + absorbed_down_in_ec_2 + absorbed_out_ec_2 + absorbed_photons_by_lum_2_from_emission_lum_1 + absorbed_photons_from_emission_by_lum3_in_lum2 
        total_absorbed_3 = absorbed_up_in_ec_3 + absorbed_down_in_ec_3 + absorbed_out_ec_3 + absorbed_photons_by_lum3_from_emission_by_lum1_and_2
    
        # Update the next round of emissions
        emitted_photons_list_1.append(total_absorbed_1.sum() * quantum_yield_1)
        emitted_photons_list_2.append(total_absorbed_2.sum() * quantum_yield_2)
        emitted_photons_list_3.append(total_absorbed_3.sum() * quantum_yield_3)


    # Calculate transmission fraction for the luminophore and waveguide combined
    transmission_from_emission = np.sum(photons_transmitted_through_bottom_after_ems_by_lum1_and_2, axis=0) + np.sum(photons_transmitted_through_bottom_after_ems_by_lum3, axis=0)
    transmission_spectrum = initial_transmission + transmission_from_emission
   
    # AVT Calculation
    avt = calculate_avt(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)

    # CRI Calculation
    transmission_spectrum = (speed_of_light * planck_constant * transmission_spectrum) / (light_spectrum.wavelength * 1e-9) / interval

    try:
        cri = calculate_cri(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)
    except ZeroDivisionError:
        print("ZeroDivisionerror in CRI calculation, setting at 0.0")
        cri = 0.0
          
    side_incident_photons_total = np.sum(photons_reaching_the_edge_1, axis=0) + np.sum(photons_reaching_the_edge_2, axis=0) + np.sum(photons_reaching_the_edge_3, axis=0)
   
    pce = calculate_pce(wavelengths,
                        side_incident_photons_total,
                        spectrum_start,
                        spectrum_end,
                        interval)
          
    return pce, avt, cri

def numerical_multiple_in_one(luminophore_1, concentration_1, luminophore_2, concentration_2, luminophore_3, concentration_3, parameter_dict):

     # Set start and end of wavelength distribution range (works for light, but also luminophore abs and ems limits)
     spectrum_start = parameter_dict["spectrum_start"]
     spectrum_end = parameter_dict["spectrum_end"]
     interval = parameter_dict["interval"] 
     
     # Import light spectrum, absorption, and emission using the combined function
     light_spectrum = import_spectrum("AM1.5G", spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval, plotting = False)
     
     absorption_df_1 = import_spectrum("absorption", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
     emission_df_1 = import_spectrum("emission", name=luminophore_1, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
     
     # Handle quantum yield
     if parameter_dict.get("quantum_yield_unity", False):
         quantum_yield_1 = 1.0
     else:
         quantum_yield_1 = import_quantum_yield(luminophore_1)
     
     # Import light spectrum, absorption, and emission using the combined function
     absorption_df_2 = import_spectrum("absorption", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
     emission_df_2 = import_spectrum("emission", name=luminophore_2, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
        
     # Handle quantum yield
     if parameter_dict.get("quantum_yield_unity", False):
         quantum_yield_2 = 1.0
     else:
         quantum_yield_2 = import_quantum_yield(luminophore_2)
         
     # Import light spectrum, absorption, and emission using the combined function
     absorption_df_3 = import_spectrum("absorption", name=luminophore_3, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
     emission_df_3 = import_spectrum("emission", name=luminophore_3, spectrum_start=spectrum_start, spectrum_end=spectrum_end, interval=interval)
         
     # Handle quantum yield
     if parameter_dict.get("quantum_yield_unity", False):
          quantum_yield_3 = 1.0
     else:
          quantum_yield_3 = import_quantum_yield(luminophore_3)
         
     
     # Set LSC dimensions
     width = parameter_dict["width"]
     height = parameter_dict["height"]
     thickness = parameter_dict["thickness"] 
         
     # Calculate escape cone loss based on isotropic emission
     refractive_index_wg = parameter_dict["refractive_index_wg"]
     refractive_index_air = parameter_dict["refractive_index_air"]
     
     # From McDowall et al., (2013) Applied Optics Vol. 52, Issue 6, pp. 1230-1239 https://doi.org/10.1364/AO.52.001230
     escape_cone_loss = 1 - math.sqrt(1 - (refractive_index_air/refractive_index_wg)**2) 
     
     # Import PMMA spectrum
     pmma_coefficient = pd.read_csv(os.path.join(current_dir, "Data", "PMMA_coefficient.csv"), sep=';', decimal='.')
     
     pmma_coefficient_interpolation = p1d(pmma_coefficient.wavelength, pmma_coefficient['spectrum'])
     new_range = np.arange(spectrum_start, spectrum_end + interval, interval)
     
     # Option to set the waveguide absorption to very low values.
     factor_pmma = 1E-5 if parameter_dict.get('waveguide_absorption_off', False) else 1.0
         
     pmma_coefficient_values = pmma_coefficient_interpolation(new_range) * factor_pmma
     pmma_coefficient = pd.DataFrame({'wavelength': new_range, 'spectrum': pmma_coefficient_values})
     
     # Calculate path distribution for waveguide
     path_length_interval = 0.1
     
     k_correction_factor = calculate_correction_factor(refractive_index_air, refractive_index_wg)
     
     diagonal = np.sqrt(width**2 + height**2)
     max_path_length = diagonal * k_correction_factor
     
     # Length of array (make sure it's square)
     array_length = (spectrum_end - spectrum_start) / interval
     if array_length % 1 != 0:
         raise ValueError("Array has to be of a length which floors to zero.")
     
     photon_path_lengths = np.arange(0, max_path_length, path_length_interval)
     path_length_probabilities = [path_length_probability_distribution(r, width, height, diagonal, k_correction_factor) for r in photon_path_lengths]
     total_probability = sum(path_length_probabilities)
     
     normalized_probabilities = [p / total_probability for p in path_length_probabilities]
     
     # Ensure normalized_probabilities has two dimensions.
     normalized_probabilities = np.array(normalized_probabilities)[:, np.newaxis]
        
     
     # === INITIAL ABSORPTION CALCULATIONS FOR TANDEM LSC ===
      
     # Calculate the fresnel transmission accounting for the reflection loss at the interface of two different mediums.
     fresnel_transmission = 1 - ((refractive_index_air - refractive_index_wg) / (refractive_index_air + refractive_index_wg))**2
      
     # Convert pandas series to numpy arrays for faster calculation.
     wavelengths = light_spectrum.wavelength.values
     spectrum = light_spectrum.spectrum.values
     absorption_values_1 = absorption_df_1.absorption.values
     absorption_values_2 = absorption_df_2.absorption.values
     absorption_values_3 = absorption_df_3.absorption.values
    
      
     # Calculate the spectrum after reflection loss.
     spectrum_after_reflection = spectrum * fresnel_transmission
     
     # Calculate combined absorption and absorption fraction for the first luminophore and waveguide.
     combined_absorption = absorption_values_1 * concentration_1 +  absorption_values_2 * concentration_2 +  absorption_values_3 * concentration_3 + pmma_coefficient.spectrum.values
     total_absorption_probability = 1 - np.exp(-thickness * combined_absorption)
     absorption_luminophores_and_wg = total_absorption_probability * spectrum_after_reflection
     abs_ratio_lum_1 = absorption_values_1 * concentration_1 / combined_absorption
     abs_ratio_lum_2 = absorption_values_2 * concentration_2 / combined_absorption
     abs_ratio_lum_3 = absorption_values_3 * concentration_3 / combined_absorption

     total_absorbed_photons_init_1 = np.sum(absorption_luminophores_and_wg * abs_ratio_lum_1)
     total_absorbed_photons_init_2 = np.sum(absorption_luminophores_and_wg * abs_ratio_lum_2)
     total_absorbed_photons_init_3 = np.sum(absorption_luminophores_and_wg * abs_ratio_lum_3)
     
     initial_transmission = (spectrum_after_reflection - absorption_luminophores_and_wg) * fresnel_transmission
     
     # === CALCULATE ABSORPTION PROBABILITIES LSC 2 ===
     
     # Calculate the probability of a photon being absorbed outside the escape cone (EC).
     absorption_outside_EC = np.outer(combined_absorption,  photon_path_lengths)
     probabilities_outside_EC = np.exp(-absorption_outside_EC) * normalized_probabilities.T
     
     # Create a dataframe with probabilities of a photon reaching the edge.
     probabilities = pd.DataFrame({'probability_photon_reaching_edge': probabilities_outside_EC.sum(axis=1),
                                     'wavelength': light_spectrum.wavelength})
     
     
     # Calculate absorption probabilities within the escape cone.
     critical_angle = math.asin(refractive_index_air/refractive_index_wg)
     k_correction_in_escape_cone = calculate_correction_factor(1.0, 1/np.cos(critical_angle))
     half_total_absorption = 0.5 * (1 - np.exp(-thickness * combined_absorption))
     depth_half_absorbed = -np.log(1 - half_total_absorption) / combined_absorption
     
     probabilities['absorption_probs_in_ec_up'] = 1 - np.exp(-depth_half_absorbed * k_correction_in_escape_cone * combined_absorption)
     probabilities['absorption_probs_in_ec_down'] = 1 - np.exp(-(thickness - depth_half_absorbed) * k_correction_in_escape_cone * combined_absorption)
     
     # Normalize emission spectrum
     probabilities['normalized_emission_1'] = emission_df_1.emission / emission_df_1.emission.sum()
     probabilities['normalized_emission_2'] = emission_df_2.emission / emission_df_2.emission.sum()
     probabilities['normalized_emission_3'] = emission_df_3.emission / emission_df_3.emission.sum()

     
     
     # === CALCULATE EMISSION EVENTS ===
      
     # Set the number of re-emissions to be considered.
     number_of_re_emissions = 5
      
     # Initialize variables to store results
     emitted_photons_list_1 = [total_absorbed_photons_init_1 * quantum_yield_1]
     emitted_photons_list_2 = [total_absorbed_photons_init_2 * quantum_yield_2]
     emitted_photons_list_3 = [total_absorbed_photons_init_3 * quantum_yield_3]
     photons_reaching_the_edge = []
     photons_transmitted_through_bottom_after_ems = []
     
     # Calculate values that are used repeatedly outside the loop
     prob_photon_edge = probabilities.probability_photon_reaching_edge.values
     abs_prob_in_ec_up = probabilities.absorption_probs_in_ec_up.values
     abs_prob_in_ec_down =  probabilities.absorption_probs_in_ec_down.values
     
     # Calculate frequently occuring variables
     norm_emit_escape_1 = probabilities.normalized_emission_1.values * escape_cone_loss / 2
     norm_emit_outside_1 = probabilities.normalized_emission_1.values * (1 - escape_cone_loss)
     norm_emit_escape_2 = probabilities.normalized_emission_2.values * escape_cone_loss / 2
     norm_emit_outside_2 = probabilities.normalized_emission_2.values * (1 - escape_cone_loss)
     norm_emit_escape_3 = probabilities.normalized_emission_3.values * escape_cone_loss / 2
     norm_emit_outside_3 = probabilities.normalized_emission_3.values * (1 - escape_cone_loss)
    
     for i in range(number_of_re_emissions):
         
        emitted_photons_current_1 = emitted_photons_list_1[-1]  # Get the last value
        emitted_photons_current_2 = emitted_photons_list_2[-1]  # Get the last value
        emitted_photons_current_3 = emitted_photons_list_3[-1]  # Get the last value
    
    
        # Calculations for emitted photons in LSC 1
        emitted_up_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
        emitted_down_in_ec_1 = emitted_photons_current_1 * norm_emit_escape_1
        emitted_out_ec_1 = emitted_photons_current_1 * norm_emit_outside_1
        
        # Photons reaching the edge LSC 1
        reaching_edge_1 = prob_photon_edge * emitted_out_ec_1
        photons_reaching_the_edge.append(reaching_edge_1)
        
        # Calculations for emitted photons in LSC 2
        emitted_up_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
        emitted_down_in_ec_2 = emitted_photons_current_2 * norm_emit_escape_2
        emitted_out_ec_2 = emitted_photons_current_2 * norm_emit_outside_2
        
        # Photons reaching the edge LSC 2
        reaching_edge_2 = prob_photon_edge * emitted_out_ec_2
        photons_reaching_the_edge.append(reaching_edge_2)
        
        # Calculations for emitted photons in LSC 3
        emitted_up_in_ec_3 = emitted_photons_current_3 * norm_emit_escape_3
        emitted_down_in_ec_3 = emitted_photons_current_3 * norm_emit_escape_3
        emitted_out_ec_3 = emitted_photons_current_3 * norm_emit_outside_3
        
        # Photons reaching the edge LSC 3
        reaching_edge_3 = prob_photon_edge * emitted_out_ec_3
        photons_reaching_the_edge.append(reaching_edge_3)
    
        # Calculate absorbed photons LSC 1
        absorbed_up_in_ec_1 = abs_prob_in_ec_up * emitted_up_in_ec_1  * abs_ratio_lum_1
        absorbed_down_in_ec_1 = abs_prob_in_ec_down * emitted_down_in_ec_1  * abs_ratio_lum_1
        absorbed_out_ec_1 = (1 - prob_photon_edge) * emitted_out_ec_1 * abs_ratio_lum_1
        
        # Calculate absorbed photons LSC 2
        absorbed_up_in_ec_2 = abs_prob_in_ec_up * emitted_up_in_ec_2  * abs_ratio_lum_2
        absorbed_down_in_ec_2 = abs_prob_in_ec_down * emitted_down_in_ec_2  * abs_ratio_lum_2
        absorbed_out_ec_2 = (1 - prob_photon_edge) * emitted_out_ec_2 * abs_ratio_lum_2
    
        # Calculate absorbed photons LSC 3
        absorbed_up_in_ec_3 = abs_prob_in_ec_up * emitted_up_in_ec_3  * abs_ratio_lum_3
        absorbed_down_in_ec_3 = abs_prob_in_ec_down * emitted_down_in_ec_3  * abs_ratio_lum_3
        absorbed_out_ec_3 = (1 - prob_photon_edge) * emitted_out_ec_3 * abs_ratio_lum_3
    
        # Photons transmitted through bottom after emission
        photons_transmitted_through_bottom_after_ems.append((1 -  abs_prob_in_ec_down) * (emitted_down_in_ec_1 + emitted_down_in_ec_2 + emitted_down_in_ec_3))
        
        # Total absorbed photons for luminophore
        total_absorbed_1 = absorbed_up_in_ec_1 + absorbed_down_in_ec_1 + absorbed_out_ec_1 
        total_absorbed_2 = absorbed_up_in_ec_2 + absorbed_down_in_ec_2 + absorbed_out_ec_2
        total_absorbed_3 = absorbed_up_in_ec_3 + absorbed_down_in_ec_3 + absorbed_out_ec_3 
    
        # Update the next round of emissions
        emitted_photons_list_1.append(total_absorbed_1.sum() * quantum_yield_1)
        emitted_photons_list_2.append(total_absorbed_2.sum() * quantum_yield_2)
        emitted_photons_list_3.append(total_absorbed_3.sum() * quantum_yield_3)
    
    
     # Calculate transmission fraction for the luminophore and waveguide combined
     transmission_from_emission = np.sum(photons_transmitted_through_bottom_after_ems, axis=0)
     transmission_spectrum = initial_transmission + transmission_from_emission
       
     # Calculate transmission fraction for the luminophore and waveguide combined
     transmission_from_emission = np.sum(photons_transmitted_through_bottom_after_ems, axis=0)
     transmission_spectrum = initial_transmission + transmission_from_emission
       
     # AVT Calculation
     avt = calculate_avt(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)
    
     # CRI Calculation
     transmission_spectrum = (speed_of_light * planck_constant * transmission_spectrum) / (light_spectrum.wavelength * 1e-9) / interval
    
     try:
         cri = calculate_cri(wavelengths_of_spectrum = light_spectrum.wavelength.values, transmitted_spectrum = transmission_spectrum, interval = interval)
     except ZeroDivisionError:
        print("ZeroDivisionerror in CRI calculation, setting at 0.0")
        cri = 0.0
          
     side_incident_photons_total = np.sum(photons_reaching_the_edge, axis=0)
       
     pce = calculate_pce(wavelengths,
                        side_incident_photons_total,
                        spectrum_start,
                        spectrum_end,
                        interval)    
    
    
    
     return pce, avt, cri
