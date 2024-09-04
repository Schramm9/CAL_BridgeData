# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:13:06 2024

@author: Chris
"""

# What is the overall procedure to get the data from a single column (CS1 most likely) and split it off from the original dataframe inside the dict, most likely into another dict wherein the rest of the dataframes will be sliced off similarly-

# Starting from the element_dfs dict, make the new dict of dfs for the CS1 column: element_dfs_CS1 

# Then copy off the year column rather than slicing it off from the df inside the dict. So element_dfs_CS1 (+ year column)

# Then remove any zeros and 1s as you may think necessary, do that by making it into a separate dict zeros_ones_CS1 (i.e. similar to element_dfs.copy()) after making the zeros_ones_CS1 dict make another copy of that and remove the outliers from it: zeros_ones_CS1_no_outls

# Then take the "original" element_dfs_CS1 (pre remove 1s and zeros) and remove the outliers only: element_dfs_CS1_no_outls 

# I think this is the extent of what I want to do pre regression. So  make the plots with the dicts of dataframes element_dfs_CS1 (which is the CS1 column sliced off from element_dfs with the CS1 column copied from the original)

# First element_dfs_CS1
# Then zeros_ones_CS1
# Then zeros_ones_CS1_no_outls
# And lastly element_dfs_CS1_no_outls (or dict_w_outls_rmvd)


#def change_string(input_string:str) -> None:
""" Notice that this function doesn't return anything! """
 #   input_string += 'a'

#def change_list(input_list:list) -> None:
""" Notice that this function doesn't return anything! """
 #   input_list.append('a')

"""
# New dictionary to store DataFrames with removed column
element_dfs_CS1 = {} # new_data

# Loop through the original dictionary of DataFrames
for df_keyname, df in element_dfs.items():  # df_name df data.items
    # Create a new DataFrame with the removed columns
    cs1_col_df = df.drop(columns=rem_cols) # new_df
    
    # Store the new DataFrame in the new dictionary
    element_dfs_CS1[df_keyname] = cs1_col_df

"""





# Begin removing 1s from the element_dfs_CS1 dictionary procedure

def rmv_ones_for_CS1(original_dict, col_names):
    mod_dict = {}  # holds the modified dataframes where rows holding values of 1 in the CS1 column are removed.

    for key, dataframe in original_dict.items():
        mod_df = dataframe.copy()  # copy of the original df to remove the 1s
        
        for col_name in col_names:
            if col_name in mod_df.columns:
                mask = mod_df[col_name] == 1
                mod_df.drop(mod_df[mask].index, inplace=True)
        
        mod_dict[key] = mod_df  # Store the modified DataFrame in the new dictionary
    
    return mod_dict

col_names = ['CS1']


# element_dfs_CS1_1s represents the dictionary of dataframes with the ones removed from the CS1 column.  
element_dfs_CS1_1s = rmv_ones_for_CS1(element_dfs_CS1, col_names)

# element_dfs_CS1_1s for df abmt_rc_215 length post rmv_ones_for_CS1 is 19206 rows

# End Remove rows with 1s (ones) from the dataframes based on the presence of 1s in CS1




# I think the data will produce better results if the rows with zeros in  the CS1 column are removed as well.  

# Begin Remove rows with zeros (0) from the dataframes based on the presence of zeros in CS1

def rmv_zeros_for_CS1(original_dict, col_names):
    mod_dict = {}  # holds the modified dataframes with the rows holding zeros in CS1 column removed

    for key, dataframe in original_dict.items():
        mod_df = dataframe.copy() # copy of oringinal to remove the zeros from
        
        for col_name in col_names:
            if col_name in mod_df.columns:
                mask = mod_df[col_name] == 0
                mod_df.drop(mod_df[mask].index, inplace=True)
        
        mod_dict[key] = mod_df  # Store the modified DataFrame in the new dictionary
    
    return mod_dict

col_names = ['CS1']


# element_dfs_CS1_0s represents the dictionary of dataframes with the zeros removed from the CS1 column.  
element_dfs_CS1_0s = rmv_zeros_for_CS1(element_dfs_CS1, col_names)

# End Remove rows with zeros (0) from the dataframes based on the presence of zeros in CS1



# Begin remove columns from element_dfs except for CS1

# !!!
# List of columns to be removed and saved in new, different DataFrames- ORIGINALLY rem_cols = ['CS2', 'CS3', 'CS4'] going to switch it to CS3 and CS4 only.  


# !!!
rem_cols = ['CS3', 'CS4'] # columns_to_remove

# New dictionary to store DataFrames with removed column
element_dfs_CS1 = {} # new_data with the CS3 and 4 columns removed ORIGINALLY CS2 was also removed.  Still going to stick with calling it _CS1 suffix- for now. 

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

# End remove columns from element_dfs except for CS1 AND CS2


"""
print(geo_df.dtypes)

print(geo_df)

print("Coordinates GeoDataFrame CRS:", geo_df.crs)

if geo_df.crs != state_gdf.crs:
    geo_df = geo_df.to_crs(state_gdf.crs)

print("State GeoDataFrame bounds:", state_gdf.total_bounds)
print("Coordinates GeoDataFrame bounds:", geo_df.total_bounds)

# Line 1714: ValueError: The GeoSeries already has a CRS which is not equal to the passed CRS. Specify 'allow_override=True' to allow replacing the existing CRS without doing any transformation. If you actually want to transform the geometries, use 'GeoSeries.to_crs' instead.

#!!!
# Still not getting the boundary and the points shown on the same plot

geo_df.set_crs('EPSG:4326', inplace=True)  # Set CRS for the points, e.g., WGS84
geo_df.to_crs(state_gdf.crs, inplace=True)  # Transform to match the state's CRS
"""