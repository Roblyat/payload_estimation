#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

def process_joint_states(joint_states_csv, output_csv):
    # Load the joint states CSV file
    df_joint_states = pd.read_csv(joint_states_csv)

    # Step 1: Remove unnecessary joints
    excluded_joints = [
        'gripper_finger_joint', 
        'gripper_robotiq_85_left_finger_tip_joint', 
        'gripper_robotiq_85_left_inner_knuckle_joint',
        'gripper_robotiq_85_right_finger_tip_joint',
        'gripper_robotiq_85_right_inner_knuckle_joint',
        'gripper_robotiq_85_right_knuckle_joint',
        'sensor_measurment_joint'
    ]

    # Filter out excluded joints
    df_joint_states_filtered = df_joint_states.loc[~df_joint_states['Joint Name'].isin(excluded_joints)].copy()

    # Step 2: Group joint states by timestamp (block them by 'Time')
    grouped_joint_states = df_joint_states_filtered.groupby('Time')

    # Prepare the final DataFrame to store joint states with computed acceleration
    combined_data = []

    # Step 3: Compute the acceleration for each row based on the velocity difference between time steps
    time_step = 1 / 40  # Assuming the time step is 0.025 seconds (40Hz)

    # Iterate over the grouped timestamps and joints
    for time, group in grouped_joint_states:
        # Sort by Joint Name to ensure consistency
        group = group.sort_values(by='Joint Name').reset_index(drop=True)
        
        # For each joint at this timestamp, calculate the acceleration (velocity difference)
        for i in range(len(group)):
            if i > 0:  # Skip the first row (as there is no previous time to compare)
                acceleration = (group.iloc[i]['Velocity'] - group.iloc[i-1]['Velocity']) / time_step
            else:
                acceleration = group.iloc[i]['Velocity'] / time_step  # For the first entry, use velocity/time

            # Append the new data (with calculated acceleration)
            combined_data.append([
                group.iloc[i]['Time'],           # Time
                group.iloc[i]['Joint Name'],     # Joint Name
                group.iloc[i]['Position'],       # Position
                group.iloc[i]['Velocity'],       # Velocity
                acceleration,                    # Calculated Acceleration
                group.iloc[i]['Effort']          # Effort
            ])

    # Step 4: Define column names for the new CSV
    combined_columns = ['Time', 'Joint Name', 'Position', 'Velocity', 'Acceleration', 'Effort']

    # Step 5: Create a DataFrame from the combined data and save it to CSV
    df_combined = pd.DataFrame(combined_data, columns=combined_columns)
    df_combined.to_csv(output_csv, index=False)

    print(f"Joint states with computed accelerations saved to {output_csv}")

if __name__ == "__main__":
    
    # Paths to CSV files
    joint_states_csv = "/workspace/shared/data/raw/0_raw_data.csv"
    combined_csv = "/workspace/shared/data/processed/raw_data.csv"

    # Step 1: Process joint states and wrench data
    process_joint_states(joint_states_csv, combined_csv)