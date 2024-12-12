"""
                             DISCLAIMER:
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
"""

import os
import pandas as pd
import hyperspy.api as hs #v1.7.6
import numpy as np

def extract_kfactors(kfactor_lines):
    
    """
    Processes a K-factor data file exported from Velox and converts it 
    into a pandas DataFrame suitable for EDS quantification.

    Args:
        kfactor_lines (str): Path to the CSV file containing K-factor data.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            * Element: The element associated with the K-factor.
            * Line: The X-ray line identifier.
            * Name: The original identifier from the K-factor file.
            * HS_Reference: The HyperSpy reference name for the X-ray line.
            * K-factor: The normalized K-factor value (normalized by Si-Ka1).

    Notes:
        * This function assumes the K-factor data file is formatted as exported from Velox.
        * The code was originally developed by Thomas Aarholt and adapted for this project.
        * Git repository: https://gist.github.com/thomasaarholt/1f44648e5f9adf3cfd9211dc0e492d4b/revisions
    """

    df = pd.read_csv(kfactor_lines)

    DF = df.iloc[1:] # remove empty row in dataset
    DF.loc[:,'K-factor'] = DF['K-factor'].astype(float) # String to float on the k-factors

    # Two functions that we map across the dataset to split the header into separate elements and line
    def splitelement(entry):
        return entry.split("-")[0]
    def splitline(entry):
        return entry.split("-")[1]

    DF.loc[:,'Element']  = DF['Line identifier'].map(splitelement)
    DF.loc[:, 'Line']  = DF['Line identifier'].map(splitline)
    DF.loc[:, 'Name']  = DF['Line identifier']
    si_k_factor = DF['K-factor'][DF['Name'] == 'Si-Ka1']
    DF.loc[:, 'K-factor'] = DF['K-factor'] / si_k_factor.values # It's normalized by top element. We want by Si Ka1
    DF['HS_Reference'] = DF['Name'].str.replace('-','_').str.replace('Ka1','Ka').str.replace('La1','La').str.replace('Ma1','Ma')# Add HD reference column and rename x-ray line names to match HS format
    kfactor_DF = DF[['Element', 'Line', 'Name', 'HS_Reference', 'K-factor']] # This is the new order of columns

    return kfactor_DF


def EDS_chipping(EDS_file, kfactor_path, scan, crop, crop_string, chip_size, 
                 additional_elements = ["Cu", "C", "N", "Al", "Ga"],
                 xray_lines = ['Al_Ka', 'C_Ka', 'Cu_Ka', 'Fe_Ka', 'Ga_Ka', 'La_La', 'N_Ka', 'O_Ka', 'Pt_La', 'Sr_Ka', 'Ti_Ka']):
    
    """
    Processes an EDS data file (TEM) and performs chipping and quantification.

    Args:
        EDS_file (str): Path to the EDS data file (.emd format).
        kfactor_path (str): Path to the K-factor data file (CSV format).
        scan (str): Scan number associated with the EDS sample.
        crop (bool): Flag indicating whether to crop the data.
        crop_string (str): String defining the crop region in format "ymin:ymax,xmin:xmax".
        chip_size (int): Size of each chipped region.
        additional_elements (lists): List of elements to analyze.
        xray_lines (list): List of X-ray lines to analyze.

    Returns:
        tuple: A tuple containing two DataFrames:
            * eds_spec_df (pandas.DataFrame): DataFrame containing chip IDs and corresponding EDS spectra.
            * atom_percent_df (pandas.DataFrame): DataFrame containing chip IDs and quantified atomic percent for each element.

    """

    # Define signal trim value
    isig_trim = '0.2:'
    
    # Define K-factors
    try:
        # Extract K-factors
        kfactor_DF = extract_kfactors(kfactor_path)
        
        # Check if the required columns are present
        required_columns = ['HS_Reference', 'K-factor']
        if all(column in kfactor_DF.columns for column in required_columns):
            print("Both 'HS_Reference' and 'K-factor' columns are present in the k-factor data frame.")
        else:
            missing_columns = [column for column in required_columns if column not in kfactor_DF.columns]
            print(f"The following required columns are missing from the k-factor data frame: {', '.join(missing_columns)}")
    except:
        print("Unable to extract K-factors")
    else:
        print("K-factors successfully extracted")

    ## Load and Clean EDS data
    s = hs.load(EDS_file, signal_type="EDS_TEM")
    
    # Ensure EDS_TEM spectrum has two dimensions:
    for item in s:
        if item.axes_manager.signal_indices_in_array == (2,):
            EDS_spectrum = item
            break
    else:
        EDS_spectrum = None
        print("EDS_TEM spectrum with correct dimensions cannot be found in selected file")
        return None, None
    
    # Add elements/x-ray lines:
    EDS_spectrum.add_elements(additional_elements) # Add known "contamination" elements
    EDS_spectrum.set_lines(xray_lines) # Define X-ray lines in metadata   
    
    # Crop spectra:
    EDS_spectrum = EDS_spectrum.isig[isig_trim] # Trim the signal data
    
    if crop == True:
        xmin = str(crop_string).split(',')[1].split(':')[0]
        xmax = str(crop_string).split(',')[1].split(':')[1]
        ymin = str(crop_string).split(',')[0].split(':')[0]
        ymax = str(crop_string).split(',')[0].split(':')[1]
        EDS_spectrum = EDS_spectrum.inav[eval(xmin):eval(xmax), eval(ymin):eval(ymax)] # Crop the navigational data (left, right, bottom, and top)
    
    # Re-bin data:
    EDS_spectrum.axes_manager[-1].is_binned = True
    EDS_spectrum = EDS_spectrum.rebin(scale=[chip_size, chip_size, 1])



    ## Chip EDS data
    # Create a data frame of chip IDs and EDS spectra:
    rows, cols = EDS_spectrum.axes_manager[1].size, EDS_spectrum.axes_manager[0].size
    eds_spec_dict = {'Chip ID': [], 'eds spectra': []}
    
    Chip_ID_List = []
    
    for row in range(rows): 
        for col in range(cols):
            chip_ID = f'R{row}C{col}'  # Create chip ID
            Chip_ID_List.append(chip_ID)
            eds_spectrum_data = EDS_spectrum.inav[col, row].data  # Extract chip's spectrum data
            eds_spec_dict['Chip ID'].append(chip_ID)
            eds_spec_dict['eds spectra'].append(eds_spectrum_data)
    eds_spec_df = pd.DataFrame(eds_spec_dict)
    
    # Save using numpy savez (.npz):
    file_path_npz = os.path.join(f"Extracted_Spectra_{os.path.basename(EDS_file).replace('/', '-').replace('.emd', '')}.npz")
    np.savez(file_path_npz, Chip_ID=eds_spec_dict['Chip ID'], eds_spectrum=eds_spec_dict['eds spectra']) 



    ## Quantify EDS data:
    # Define model
    model = EDS_spectrum.create_model()
    model.signal.models
    model.multifit(iterpath="flyback", bounded=True)
    intensity = model.get_lines_intensity()
    
    # Create a dictionary with matched k-factor values (or 0 if nan), from the xray_lines list:
    matched_kfactor_dict = {line: kfactor_DF.loc[kfactor_DF['HS_Reference'] == line, 'K-factor'].values[0] if line 
                            in kfactor_DF['HS_Reference'].values else 0 for line in xray_lines}
    
    # Filter intensity based on selected X-ray lines
    selected_base_signals = [signal for signal in intensity 
                             if any(line in signal.metadata.General.title.__str__() for line in list(matched_kfactor_dict.keys()))]
    
    # Calculate atomic percent
    atomic_percent = EDS_spectrum.quantification(selected_base_signals, 'CL', factors=list(matched_kfactor_dict.values()))
        
    # Create a data frame for atomic%:
    atom_percent_df = pd.DataFrame()
    atom_percent_df['Chip ID'] = Chip_ID_List
    
    for signal in atomic_percent:
        column_name = signal.metadata.General.title.replace('atomic percent of ', '')
        atom_percent_df[column_name] = signal.data.flatten()



    ## Save Results:
    atomPer_file_path = os.path.join(f"Atom Percent_{os.path.basename(EDS_file).replace('/', '-').replace('.emd', '')}.csv")
    atom_percent_df.to_csv(atomPer_file_path, index=False)
    
    return eds_spec_df, atom_percent_df
