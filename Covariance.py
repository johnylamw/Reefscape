import pandas as pd
import argparse
import numpy as np
parser = argparse.ArgumentParser(
    prog="AdvantageScope Covariance",
    description="Calculates a Covariance Matrix via Advantage Scope output data"
)
parser.add_argument("filename")

# Define groups with their respective prefixes, systems, values, and abbreviations.
groups = [
    {
        "base_prefix": "NT:/AdvantageKit/RealOutputs/PoseSubsystem",
        "systems": [
            "OdometryOnlyRobotPose",
            "RobotPose",
            "VisionEnhancedPose",
            "WheelsOnlyEstimate"
        ],
        "values": ["rotation/value", "translation/x", "translation/y"],
        "abbr": {
            "OdometryOnlyRobotPose": "O",
            "RobotPose": "R",
            "VisionEnhancedPose": "V",
            "WheelsOnlyEstimate": "W"
        }
    },
    {
        "base_prefix": "NT:/Trackers",
        "systems": [
            "Tracker_1",
            "Tracker_2",
            "Tracker_3"
        ],
        "values": ["rotation/value", "translation/x", "translation/y"],
        "abbr": {
            "Tracker_1": "T1",
            "Tracker_2": "T2",
            "Tracker_3": "T3"
        }
    },
    {
        "base_prefix": "NT:/photonvision/Apriltag_RearLeft_Camera",
        "systems": [
            "targetPose"
        ],
        "values": ["rotation/q/w", "rotation/q/x", "rotation/q/y", "rotation/q/z", "translation/x", "translation/y", "translation/z"],
        "abbr": {
            "targetPose": "TP"
        }
}
]

def compute_covariance_matrix(csv_file):
    # Build a dictionary mapping short keys to their full column names across all groups.
    column_dictionary = {}
    
    for group in groups:
        base_prefix = group["base_prefix"]
        systems = group["systems"]
        values = group["values"]
        abbr = group["abbr"]
        for system in systems:
            for value in values:
                full_col = f"{base_prefix}/{system}/{value}"
                # Use the abbreviation plus the last part of the value (e.g., "value", "x", "y") for the short key.
                #short_key = f"{abbr[system]}_{value.split('/')[-1]}"
                #short_key = f"{abbr[system]}_{value.split('/')[-2]}_{value.split('/')[-1]}"
                short_key = f"{abbr[system]}_{value}"
                column_dictionary[short_key] = full_col

    # Attempt to open the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print("Cannot read csv_file", csv_file, "with Exception:", e)
        return

    # Filter the columns that exist in the CSV using the dictionary
    available_keys = {k: v for k, v in column_dictionary.items() if v in df.columns}
    
    if not available_keys:
        print("None of the specified columns were found in the CSV file.")
        return

    # Extract the subset and rename columns using the short keys
    df_subset = df[list(available_keys.values())].rename(columns={v: k for k, v in available_keys.items()})
    
    print("Using the following mapped columns:")
    print(df_subset.columns.tolist())
    
    # Compute the covariance matrix
    cov_matrix = df_subset.cov()

    # Filtering (for readability)
    threshold = 1e-4
    cov_matrix[np.abs(cov_matrix) < threshold] = 0
    
    print("\nCovariance Matrix:")
    print(cov_matrix)
    
    # Write the covariance matrix to a CSV file
    file_name_exclude_extension = file_name.split(".")[0]
    output_file = f"{file_name_exclude_extension}_MAT.csv"
    try:
        cov_matrix.to_csv(output_file)
        print(f"\nCovariance matrix has been written to '{output_file}'.")
    except Exception as e:
        print("Failed to write covariance matrix to CSV:", e)

if __name__ == "__main__":
    args = parser.parse_args()
    file_name = args.filename
    compute_covariance_matrix(file_name)