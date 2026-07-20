import pandas as pd
import numpy as np

def read_ac_power_flow_data(file_path, maxLen= -1):
    """
    Reads AC power flow data from a CSV file and processes it.

    The function identifies columns based on their suffixes (e.g., ':v_bus', ':p_from')
    to remain flexible for different power system case files.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        tuple: A tuple containing six NumPy arrays:
            - bus_voltage_magnitude (np.ndarray): Magnitudes of bus voltages.
            - bus_voltage_angle (np.ndarray): Angles of bus voltages in radians.
            - branch_p_from (np.ndarray): Active power flow from branches.
            - branch_p_to (np.ndarray): Active power flow to branches.
            - branch_q_from (np.ndarray): Reactive power flow from branches.
            - branch_q_to (np.ndarray): Reactive power flow to branches.
    """
    try:
        # Read the entire CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # --- 1. Process Bus Voltage Data ---
        # Find all columns related to bus voltage
        v_bus_cols = [col for col in df.columns if col.endswith(':v_bus')]
        if not v_bus_cols:
            raise ValueError("No bus voltage columns (ending with ':v_bus') found.")
            
        v_bus_df = df[v_bus_cols]

        # Define a robust function to convert strings to complex numbers
        def string_to_complex(s):
            if pd.isna(s):
                return np.nan # Return NaN for missing values
            # Ensure the value is a string before stripping, then convert
            return complex(str(s).replace(" ", ""))


        # Use .map (replaces deprecated .applymap) with the robust conversion function
        v_complex = v_bus_df.map(string_to_complex).values

        # Calculate magnitude and angle
        bus_voltage_magnitude = np.abs(v_complex)
        bus_voltage_angle = -180/np.pi*np.angle(v_complex) # Angle is in radians, correction based on https://github.com/NREL/OPFLearn.jl

        # --- 2. Process Branch Power Flow Data ---
        # Helper function to extract and sort data columns
        def get_data_by_suffix(suffix):
            cols = [col for col in df.columns if col.endswith(suffix)]
            if not cols:
                # Return an empty array of the correct shape if no columns are found
                return np.empty((len(df), 0))
            return df[cols].values

        # Extract active and reactive power flow data
        branch_p_from = get_data_by_suffix(':p_fr')
        branch_p_to = get_data_by_suffix(':p_to')
        branch_q_from = get_data_by_suffix(':q_fr')
        branch_q_to = get_data_by_suffix(':q_to')

        return (bus_voltage_magnitude[:maxLen, :], bus_voltage_angle[:maxLen, :], 
                branch_p_from[:maxLen, :], branch_p_to[:maxLen, :], 
                branch_q_from[:maxLen, :], branch_q_to[:maxLen, :])

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return (None,) * 6
    except Exception as e:
        print(f"An error occurred: {e}")
        return (None,) * 6
    
def read_load_gen_data(file_path, maxLen= -1):
    """
    Reads AC power flow data from a CSV file and processes it.

    The function identifies columns based on their suffixes (e.g., ':v_bus', ':p_from')
    to remain flexible for different power system case files.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        tuple: A tuple containing six NumPy arrays:
            - bus_voltage_magnitude (np.ndarray): Magnitudes of bus voltages.
            - bus_voltage_angle (np.ndarray): Angles of bus voltages in radians.
            - branch_p_from (np.ndarray): Active power flow from branches.
            - branch_p_to (np.ndarray): Active power flow to branches.
            - branch_q_from (np.ndarray): Reactive power flow from branches.
            - branch_q_to (np.ndarray): Reactive power flow to branches.
    """
    try:
        # Read the entire CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # --- 1. Process Bus Voltage Data ---
        # Find all columns related to bus voltage
        v_bus_cols = [col for col in df.columns if col.endswith(':v_bus')]
        if not v_bus_cols:
            raise ValueError("No bus voltage columns (ending with ':v_bus') found.")
            
        v_bus_df = df[v_bus_cols]

        # Define a robust function to convert strings to complex numbers
        def string_to_complex(s):
            if pd.isna(s):
                return np.nan # Return NaN for missing values
            # Ensure the value is a string before stripping, then convert
            return complex(str(s).replace(" ", ""))


        # Use .map (replaces deprecated .applymap) with the robust conversion function
        v_complex = v_bus_df.map(string_to_complex).values

        # Calculate magnitude and angle
        bus_voltage_magnitude = np.abs(v_complex)
        bus_voltage_angle = -180/np.pi*np.angle(v_complex) # Angle is in radians, correction based on https://github.com/NREL/OPFLearn.jl

        # --- 2. Process Branch Power Flow Data ---
        # Helper function to extract and sort data columns
        def get_data_by_suffix(suffix):
            cols = [col for col in df.columns if col.endswith(suffix)]
            if not cols:
                # Return an empty array of the correct shape if no columns are found
                return np.empty((len(df), 0))
            return df[cols].values

        # Extract active and reactive power flow data
        loadP = get_data_by_suffix(':pl')*10
        loadQ = get_data_by_suffix(':ql')*10

        return (loadP[:maxLen, :], loadQ[:maxLen, :])

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return (None,) * 6
    except Exception as e:
        print(f"An error occurred: {e}")
        return (None,) * 6