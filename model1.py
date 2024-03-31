# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:45:08 2024

@author: Chris
"""

# Begin remove columns from element_dfs except for CS1

def rem_cols_mk_df_model1(df, columns):
    

# List of columns to be removed and saved in new, different DataFrames
rem_cols = ['CS2', 'CS3', 'CS4'] # columns_to_remove

# New dictionary to store DataFrames with removed column
element_dfs_CS1 = {} # new_data with the CS2 3 and 4 columns removed.  

# look thru the original dict of dataframes
for df_keyname, df in element_dfs.items():  # df_name df data.items
    # Create a new DataFrame with the removed columns
    cs1_col_df = df.drop(columns=rem_cols) # new_df
    
    # Store the new DataFrames in the new dictionary
    element_dfs_CS1[df_keyname] = cs1_col_df

# Display the original dictionary of DataFrames
print("Original DataFrames:")
for df_keyname, df in element_dfs.items():
    print(f"\n{df_keyname}:")
    print(df)

# Display the new dictionary of DataFrames with removed columns
print("\nNew DataFrames:")
for df_keyname, df in element_dfs_CS1.items():
    print(f"\n{df_keyname}:")
    print(df)

# The original dictionary of DataFrames (element_dfs) remains unchanged.

# End remove columns from element_dfs except for CS1
