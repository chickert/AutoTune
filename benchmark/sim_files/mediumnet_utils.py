import pandas as pd

def rds_to_sumo_flowrouter(rds_filepath, output_filepath):
    """
    Function to convert TDOT RDS data (from CorridorCalibration-produced .csv) 
    into SUMO FlowRouter/DFRouter format.

    Parameters:
    - rds_filepath: str, path to the input RDS CSV file.
    - output_filepath: str, path to save the output SUMO FlowRouter CSV file.
    """
    # 1. Define the specific Detectors required (from your XML list)
    valid_detectors = [
        # 56.7 (onramp)
        '56.7_0', '56.7_1', '56.7_2', '56.7_3', '56.7_4',
        # 56.3 (5 lanes)
        '56.3_0', '56.3_1', '56.3_2', '56.3_3', '56.3_4',
        # 56.0 (haywood on-ramp)
        '56.0_0', '56.0_1', '56.0_2', '56.0_3', '56.0_4',
        # 55.3 RDS
        '55.3_0', '55.3_1', '55.3_2', '55.3_3',
        # 54.6 RDS
        '54.6_0', '54.6_1', '54.6_2', '54.6_3'
    ]

    # 2. Load Data
    df = pd.read_csv(rds_filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')

    # 3. Clean and Create IDs
    # Drop invalid lanes (i.e., NaN lanes)
    df = df.dropna(subset=['lane'])
    
    # Convert lane to int (e.g., 1.0 -> 1)
    df['lane'] = df['lane'].astype(int)

    # Create 'Detector' ID: "milemarker_laneIndex"
    # Note: milemarker is converted to str (56.0 -> "56.0")
    # Lane is converted from RDS 1-index to SUMO's 0-index (RDS 1 -> SUMO 0)
    # Note: This RDS -> SUMO conversion is specific to the detector mapping in the .add.xml file, 
    # which handles the fact that RDS lanes are indexed from the left, while SUMO lanes are indexed from the right.
    df['Detector'] = (
        df['milemarker'].astype(str) + "_" + (df['lane'] - 1).astype(str)
    )

    # 4. Filter for only the desired detectors
    df = df[df['Detector'].isin(valid_detectors)]

    # 5. Aggregate by 30-second intervals and Detector
    agg_df = df.groupby([
        pd.Grouper(key='timestamp', freq='30s'), 
        'Detector'
    ]).agg({
        'volume': 'sum',   # qPKW (summed count)
        'speed': 'mean'    # vPKW (average speed)
    }).reset_index()

    # 6. Process Columns for SUMO
    # Time: Minutes from start start of the day
    start_time = agg_df['timestamp'].iloc[0].normalize() # Beginning of the day
    print(f"start_time: {start_time}")
    agg_df['Time'] = (agg_df['timestamp'] - start_time).dt.total_seconds() / 60.0
    agg_df['Time'] = agg_df['Time'].round(1)
    # Add 0.5 minutes since each detector reading comes at end of the 30s interval
    agg_df['Time'] += 0.5

    # qPKW: Rename volume
    agg_df.rename(columns={'volume': 'qPKW'}, inplace=True)
    
    # vPKW: Convert mph -> km/h and handle NaNs
    agg_df['vPKW'] = (agg_df['speed'] * 1.60934).fillna(0).round(2)

    # 7. Format, Sort, and Export
    final_output = agg_df[['Detector', 'Time', 'qPKW', 'vPKW']]
    final_output = final_output.sort_values(by=['Time', 'Detector'])
    
    # Export with semicolon delimiter (as required by SUMO)
    final_output.to_csv(output_filepath, index=False, sep=';')
    
    return final_output


if __name__ == "__main__":
    # Example usage
    rds_filepath = '../data/RDS/I24_WB_52_60_11132023.csv' # Or path to other RDS data file
    output_filepath = './mednet_data_fullday.csv' # Output path for SUMO FlowRouter CSV
    df = rds_to_sumo_flowrouter(rds_filepath, output_filepath)
    