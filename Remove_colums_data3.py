<<<<<<< HEAD
import pandas as pd

# Read the CSV file
df = pd.read_csv('C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data3.csv')

# Drop the first column by index (0 represents the first column)
df = df.drop(df.columns[0], axis=1)
df = df.drop('mode', axis=1, errors='ignore')

# Save the modified DataFrame to a new CSV file
df.to_csv('C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data3_updated.csv', index=False)
=======
import pandas as pd

# Read the CSV file
df = pd.read_csv('C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data3.csv')

# Drop the first column by index (0 represents the first column)
df = df.drop(df.columns[0], axis=1)
df = df.drop('mode', axis=1, errors='ignore')

# Save the modified DataFrame to a new CSV file
df.to_csv('C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data3_updated.csv', index=False)
>>>>>>> 65036df0fa401b2f794301a7f65372ab463333b2
