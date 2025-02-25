import pandas as pd
import os


def get_data(folder_path="data_files") -> pd.DataFrame:
    # List to hold all data files
    file_list = []
    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, on_bad_lines="skip")  # Read each CSV file
            file_list.append(df)  # Add DataFrame to the list
    # Concatenate all data
    combined_df = pd.concat(file_list, ignore_index=True)
    # Save the combined data to a new CSV file
    combined_df.to_csv("combined_data.csv", index=False)
    # return combined dataset
    return combined_df
    # read_combined_df = pd.read_csv('combined_data.csv',low_memory=False)


# example
# print(get_data())
