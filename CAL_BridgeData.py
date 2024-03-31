# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:29:08 2021
@author: Chris
"""


import pandas as pd
    # pip install --upgrade pandas
    # pytz, tzdata, six, numpy, python-dateutil, pandas
import xml.etree.ElementTree as et
        # package elementpath
    
import collections
    # conda update conda
    # conda update python
    
import numpy as np
    # pip install --upgrade pandas
import numpy.ma as ma
        # 
import matplotlib.pyplot as plt
    # pip install --upgrade matplotlib
    # zipp, pyparsing, pillow, packaging, kiwisolver, fonttools, cycler, contourpy, importlib-resources, matplotlib
from datetime import date, datetime, timedelta
    # pip install --upgrade datetime
    # zope.interface, datetime
import datetime as dt
    
#import plotly.graph_objects as go
    
import copy
    
import io
    
import os
    
    
import functools as ft
from functools import reduce
    
import matplotlib.dates as mpl_dates
from matplotlib.dates import date2num
import seaborn as sns
    # pip install --upgrade seaborn
import time
    
    
import matplotlib.ticker as ticker
    
import statsmodels.api as sm
    # scipy, patsy, statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

import math
import warnings
    
import re

#import geopandas as gpd

import json
import plotly.graph_objects as go

#import folium


# import radar
    
"""
start_time = time.time()
"""
    
    
"""  
dir_path = "./CAL_BridgeData"

files = os.listdir(dir_path)

files_xml = [file for file in files if file.endswith(".xml")]
"""

# Read in the XML files as they were downloaded from the FHWA.  
 
    
def parse_XML(xml_file, df_cols): 
    """Thanks to Roberto Preste Author From XML to Pandas dataframes,
    from xtree ... to return out_df
    https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c
    """
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df
    
"""End Roberto Preste code From XML to Pandas dataframes,
    from xtree ... to return out_df
    https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c
"""    
pd. __version__ 

# Parse the XML files for the individual years (Important to keep the tags in the original files the same in the parse commands or the data at those locations may come in as None or NaN!)


import glob

# Search for the xml files from the FHWA in this (current) directory 

files = glob.glob('*.xml')
print(files)

# Split filename at the underscore (_) creates a year with the state abbreviation after- at least in the case of the files for the state of California.  

# Then follow on by adding df to the front of the string to act as variables names for the different DataFrames.

df_names = ["df" + i.split('_', 1)[0] for i in files]



root= "./CAL_BridgeData"
# container for the various xmls contained in the directory

# 46204 is the line in the df_data dataframe that marks the switch from 2016 to 2017- so the amount of data is large- may need to look into that further along as we look for things to make sense as far as this analysis goes.


# collect xml filenames and paths 

for dirpath, dirnames, filenames in os.walk(root):
    for file in files:
        files.append(dirpath + '\\' + file)
        
# end collection of xml filenames and paths



# MVP II is the Minimum Viable Product version II- or a program/data analysis that would supercede this one.  

# If I refer to "MVP II" and then comment out some code in that area I am referring to a functionality I have not achieved in this program and I would hope to make possible in a later version of this application were it to be updated.  
    

# Dataframe columns
# Hard coded, unfortunately, the format of the data files is not well suited to having the contents of the column headings read "automatically" from the files themselves as the tags that would make up the text of the column headings do not consistently appear within each <FHWAED> ... </FHWAED> tag set and basically there would be gaps in the <EPN> ... </EPN> data or the data connected to other columns would be misplaced when it is parsed to a dataframe.  

df_cols = "FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"


# Begin parsing of dataframes recursively
# create empty list for dataframes

dataframes = []

# append datasets into the list
for i in range(len(df_names)):
    temp_df = parse_XML(files[i], df_cols)
    dataframes.append(temp_df)  # creates a list of dataframes with the contents of the xml files read into them.
 
# place the filename from which data for the particular dataframe is read in as a new column in each dataframe.    
for z in range(len(files)):
    dataframes[z]['filename'] = files[z]


# !!!
# df
for df in dataframes:
    # Print the data types of the columns
    print("DataFrame:", df)
    print("Column Data Types:")
    print(df.dtypes)
    print()



# Begin data type conversion

# Define the desired data types for conversion
desired_data_types = {
'STATE': 'str',
'STRUCNUM': 'str',
'EN': 'str',
'TOTALQTY': 'float',
'CS1': 'float',
'CS2': 'float',
'CS3': 'float',
'CS4': 'float',
'filename': 'str'
}



# Iterate over the dataframes

# !!!
# df
for df in dataframes:
    # Convert data types of columns
    df = df.astype(desired_data_types)

    # Print the updated DataFrame with converted data types
    print(df)
    print()

# End data type conversion

# !!!


# !!!  
 
# df_nameToDF is the dictionary of dataframes as created when the df_names are matched up with the data corresponding to the xml file from which the data is read at parse.  
 
df_nameToDF = {df_names[x]:dataframes[x]for x in range(len(df_names))}

# creates the dictionary of keys in the form of the df_names list and the values in the form of the dataframes list.  In this case for the state of California the keys will take the form df2016CA, df2017CA, ... df2022CA.  


# bridge_counts_un means number of unique bridges or STRUCNUM in each df, performing this action to get a sense of the number of unique bridges in each data set.  


# !!!
# df
bridge_counts_un = {k: df.groupby('STRUCNUM') for k, df in df_nameToDF.items()}

# End determination of number of unique bridges

#df2016CA.groupby('STRUCNUM').count()
"""
df2016CA.groupby('STRUCNUM').count() 
# 7578

df2017CA.groupby('STRUCNUM').count()
# 10019

df2018CA.groupby('STRUCNUM').count()
# 10439
    
df2019CA.groupby('STRUCNUM').count()
# 10851

df2020CA.groupby('STRUCNUM').count()
# 10873

df2021CA.groupby('STRUCNUM').count()
# 10877

df2022CA.groupby('STRUCNUM').count()
# 10899

df2023CA.groupby('STRUCNUM').count()
# 10896
"""

# Begin df_nameToDF procedure, i.e. making the variable name with a nomenclature of df20XXCA where the XX is like 16 for 2016, 17 for 2017, etc.  

# creates the individual dataframes with names corresponding to df_names and the associated data in the form of a dataframe corresponding to the list called dataframes.

for k, v in df_nameToDF.items():
    vars()[k] = v  

# End df_nameToDF procedure



# Looked for a means of merging the dataframes within the dict of dataframes- without having to separate them out into individual variables- it may be possible, MVP II.


# In other words a lot of what happens below all the way to the area where the slicing of the concatenated dataframe begins is all meant to insure that the STRUCNUM for each year match each other and that random bridges with no matching bridge number in the other years and their associated data aren't being included in the data I intend to analyze.

# Inside the bridge_array_dict b_16 thru b_23 just mean "bridge number" aka STRUCNUM for the corresponding years- 2016 thru 2023, making a key that holds the list of STRUCNUM as an array for each year as it would be right after being parsed into a dataframe.  In other words the b_16 - b_23 variables will be larger in size (i.e. no. of rows) than the dataframes as will be seen below once the STRUCNUM not present in all years are removed.  

# Copy and then modify the existing dictionary using the existing keys to create new keys and copy the STRUCNUM column of the dataframe to the new dict and convert it to a numpy array.  

bridge_arr_dict = {'b_' + key[4:]: df['STRUCNUM'].to_numpy() for key, df in df_nameToDF.items()}



# !!! THIS IS SUBJECT TO CHANGE: The assumption I will use is that the data for all bridges (an individual bridge is denoted by STRUCNUM) that are common to all the years of data being analyzed (2016 - 2023 in this case) is to be considered, meaning that the data (condition states of individual bridge elements) associated with the bridges common across 8 years shall be used even if the data provided for a bridge one year is not provided for all the other years or is provided sporadically for other years (i.e. if the bridge components (EN) rated for condition state one year are not rated for all years - BUT some components of said bridge are rated for all years being considered and analyzed then those condition states for those elements can be used as part of the data).  For instance, a very common bridge component (or EN, element number) is a deck constructed of reinforced concrete, which I refer to as a variable as deck_rc, and this EN is number 12 as denoted by the Federal Highway Administration (FHWA) Specification for the National Bridge Inventory, Bridge Elements.  As such it may be that the number of observations across the years considered is not the same for that element number (EN) each year- some years may include condition states associated with that element present for that bridge  in some but not all years, but it is my intention to use as many observations as possible for as many bridge parts as possible to attempt to make the computer model accurate.  Again, this is subject to change.  


"""
df2016CA
# 56275 lines long

df2017CA
# 75574 lines long

df2018CA
# 78832 lines long

df2019CA
# 82570 lines long

df2020CA
# 83627 lines long

df2021CA
# 83933 lines long

df2022CA
# 85926 lines long

df2023CA
# 86474 lines long
"""

# Looking at how many occurrences of a single bridge occur in each year so that the total possible number of bridges being observed is accurate.  

# !!!
# Here
# !!!

# the dfs were creating more entries than were present originally due to additional entries created when EN = EPN that also had entries for CS1-CS4 data as well *** I still need to explain to myself why this is necessary in order to avoid multiple entries of the same element for a single bridge.  

""" the ...EPN.isnull() expressions below are required to merge the data properly while avoiding multiple entries of the same EN for each bridge  """

"""
column_name = 'EN'

dataframes = [df.dropna(subset=[column_name]) for df in dataframes]
"""

"""
null_checks = [df[column_name].isnull() for df in dataframes]

print(null_checks[0])  # Null checks for Year 2016 STRUCNUM
print(null_checks[1])  # Null checks for Year 2017 STRUCNUM
print(null_checks[2])  # Null checks for Year 2018 STRUCNUM
print(null_checks[3])  # Null checks for Year 2019 STRUCNUM
print(null_checks[4])  # Null checks for Year 2020 STRUCNUM
print(null_checks[5])  # Null checks for Year 2021 STRUCNUM
print(null_checks[6])  # Null checks for Year 2022 STRUCNUM
print(null_checks[7])  # Null checks for Year 2023 STRUCNUM
"""

# Come up with the max number of bridges that are common to all years (first merge??)
# Come up with the highest total number of possible EN for all STRUCNUM?

# MVP II is the Minimum Viable Product version II- or a program/data analysis that would supercede this one.  

# If I refer to "MVP II" and then comment out some code in that area I am referring to a functionality I have not achieved in this program and I would hope to make possible in a second version of this application were it to be updated.  
    
# MVP II: Apply isnull() method to the EPN column of the dataframes while the dataframes are still in a list or dictionary and before any other changes are made to the list, rather than applying isnull() as I do below to each dataframe hard-coded manually.


filtered_df_nameToDF = {df_name: df[df['EPN'].isnull()] for df_name, df in df_nameToDF.items()}

# change column suffixes to avoid confusion when merging at a later time in the program.  


# change dataframe columns by adding suffix to avoid MergeErrors in future versions and to enhance readability.

# df2016CA [index:length] (index starts at zero, length starts at 1) so index = 2 means '2' then length of 6 means 6 in this case, so 2016.

def add_suffix_to_col_hdr(dict_of_dfs, cols_to_exclude = None):
    for key, df in dict_of_dfs.items():
        # Extract a part of the key to use as a suffix
        suffix = key[2:6]  # Making the suffixes something like CS1_2016CA from what would have been CS1

        # Modify columns by adding the suffix
        df.columns = [col + '_' + suffix if col not in cols_to_exclude else col for col in df.columns]

# Call the function to add suffixes to columns
add_suffix_to_col_hdr(filtered_df_nameToDF, cols_to_exclude = ['STRUCNUM', 'EN'])

# Print modified dataframes
for key, df in filtered_df_nameToDF.items():
    print(f"{key}:\n{df}")


# 12.24.2023 revisions good to above.
#!!!
"""
df2016CA=df2016CA[df2016CA.EPN.isnull()]
# Drops the number of lines from 56275 to 50426

df2017CA=df2017CA[df2017CA.EPN.isnull()]
# Drops the number of lines from 75574 to 67593

df2018CA=df2018CA[df2018CA.EPN.isnull()]
# Drops the number of lines from 78832 to 70351

df2019CA=df2019CA[df2019CA.EPN.isnull()]
# Drops the number of lines from 82570 to 73470

df2020CA=df2020CA[df2020CA.EPN.isnull()]
# Drops the number of lines from 83627 to 74275

df2021CA=df2021CA[df2021CA.EPN.isnull()]
# Drops the number of lines from 83933 to 74585

df2022CA=df2022CA[df2022CA.EPN.isnull()]
# Drops the number of lines from 85926 to 75687

df2023CA=df2023CA[df2023CA.EPN.isnull()]
# Drops the number of lines from 86474 to 76053
"""


# Begin determine the set of STRUCNUM common to all years observed.  i.e, strucnum_in_all.  

strucnum_in_all = set.intersection(*map(set, bridge_arr_dict.values()))

"""
strucnum_in_all = list(set.intersection(*map(set, [b_16, b_17, b_18, b_19, b_20, b_21, b_22, b_23])))
# Sort the list so its contents will look more familiar to the user, i.e. be in numerical order.
strucnum_in_all = sorted(strucnum_in_all)
# Results in 9829 bridges starting with STRUCNUM = 01 0002 & ending with STRUCNUM = 58C0026.
"""
# End determine set of strucnum_in_all 


# BEGIN Remove STRUCNUM not present in all dfs

for df_name, df in filtered_df_nameToDF.items():
    mask = df['STRUCNUM'].isin(strucnum_in_all)
    
    df_filtered = df[mask]
   
    filtered_df_nameToDF[df_name] = df_filtered
    
# END Remove STRUCNUM not present in all dfs
    
"""

df2016CA = df2016CA[np.isin(df2016CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# No. of lines 48798

df2017CA = df2017CA[np.isin(df2017CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# No. of lines 49077

df2018CA = df2018CA[np.isin(df2018CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 57830 lines long orig.
# No. of lines 49430

df2019CA = df2019CA[np.isin(df2019CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 52036 lines long orig.
# No. of lines 49671

df2020CA = df2020CA[np.isin(df2020CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 52619 lines long orig.
# No. of lines 50137

df2021CA = df2021CA[np.isin(df2021CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 48562 lines long orig.
# No. of lines 50287

df2022CA = df2022CA[np.isin(df2022CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 48562 lines long orig.
# No. of lines 51033

df2023CA = df2023CA[np.isin(df2023CA['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 48562 lines long orig.
# No. of lines 50929
"""

# Then to run a few checks, I make the STRUCNUM into sets for each newly modified dataframe strucnum_2016_mod, strucnum_2017_mod, etc.
# !!!
# Here
# !!!

# df_names = ["df" + i.split('_', 1)[0] for i in files]


strucnum_mod = {'strucnum_' + key[2:] + '_mod': df['STRUCNUM'].unique() for key, df in filtered_df_nameToDF.items()}

"""
strucnum_2016_mod = df2016CA['STRUCNUM'].unique()

strucnum_2017_mod = df2017CA['STRUCNUM'].unique()

strucnum_2018_mod = df2018CA['STRUCNUM'].unique()

strucnum_2019_mod = df2019CA['STRUCNUM'].unique()

strucnum_2020_mod = df2020CA['STRUCNUM'].unique()

strucnum_2021_mod = df2021CA['STRUCNUM'].unique()

strucnum_2022_mod = df2022CA['STRUCNUM'].unique()

strucnum_2023_mod = df2023CA['STRUCNUM'].unique()
"""


# strucnum_2017_mod = strucnum_2018_mod = strucnum_2019_mod = strucnum_2021_mod = strucnum_2020_mod = strucnum_2021_mod = strucnum_2022_mod


    
from array import *
    
# make lists of each set of strucnum to compare them.
    
strucnum_mod = [[value] for value in strucnum_mod.values()]
    
"""
strucnum_2016_mod.tolist()

strucnum_2017_mod.tolist()
    
strucnum_2018_mod.tolist()
    
strucnum_2019_mod.tolist()
    
strucnum_2020_mod.tolist()
    
strucnum_2021_mod.tolist()
    
strucnum_2022_mod.tolist()
    
strucnum_2023_mod.tolist()
"""
    
    
# check that the STRUCNUM being pulled from each year are the same
    
# Is this the list of bridges common to all years that I would use to make the dfs that have the missing data I could replace?  MVP II.
"""
if collections.Counter(strucnum_2022_mod) == collections.Counter(strucnum_2021_mod):
        print ("The lists are identical")
    else :
        print ("The lists are not identical")
    """
    
#!!!
# 12-14-2023!!!
    
# BEGIN check content of the arrays of different years for  bridges (strucnum) in each year procedure.  
# Specifically, check the STRUCNUM in each array match each other - checking that the sets of bridges being separated out from the raw data for each year are the same.
    
def check_array_content(*arrays):
    if not arrays or len(arrays) < 2:
        return True  # If there are no or only one list, they are considered equal
    
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                if not np.array_equal(arrays[i], arrays[j]):
                    return False
    
        return True
    
    arrays_to_compare = [strucnum_mod]
    
    """[strucnum_2016_mod, strucnum_2017_mod, strucnum_2018_mod, strucnum_2019_mod, strucnum_2020_mod, strucnum_2021_mod, strucnum_2022_mod, strucnum_2023_mod]"""
    
    if check_array_content(*arrays_to_compare):
        print("All arrays are equal.")
    else:
        print("Arrays are not equal.")
    
# END check content of the arrays of different bridges (strucnum) in each year procedure.  
    
# lists associated with df2022CA and df2021CA are identical.
    
# !!!
# qty_obs_2017 = {k : f'{v}_2017' for k, v in el_names.items()} ... Make the STRUCNUM of the list strucnum_in_all into individual variables (there will be 8847 of them I assume) using the code at the beginning of this line as a guide to come up with a variable that will have the nomenclature eN_in_sTrucnum_000000000000004 (the one shown here would be a variable for STRUCNUM = 000000000000004 without anything attached to it denoting the year assessed) as a means of storing the all EN associated with that STRUCNUM for each year so there will be 8847*5 = 44235 total of those variables 
    
    
    
# I think I need to scrap all of that and just find a method that will make the set of bridges that can merge on STRUCNUM and EN obvious and therefore not be forced to come up with a means of removing individual or multiple excess lines of data from each year's individual df after coming up with a list of common EN per STRUCNUM across all years being assessed!!!!
    
    
    
    
# !!!
    
# Is this the place where I go and make all the necessary dfs for each EN PER YEAR and then follow that with code to add the time component for each individual year so as to avoid having to ensure the number of observations per year is the same?  THe answer to that question is yes.  
    
# Create EN dataframes for each year prior to concatenation of those dataframes to facilitate inserting the time component individually for each bridge element for that year based on number of observations made that year.  
    
# !!! Of course the comment above is making me wonder if the idea that I should at least be able to infer that the same amount of time has passed between successive observations of said bridge element (EN) may cause inaccuracy which would be the case if the observations of a part of a bridge were made over successive years (e.g. if STRUCNUM = 000000000000004 is observed in all years being considered but a condition state for EN = 12 IS NOT recorded each year) it would be difficult to say that the same amount of time has passed between each observation of the bridge element (EN).  
    
# !!! Going to change the protocol for this analysis as follows:
    
#  a.	IF a bridge (STRUCNUM) is not present in all years being considered that STRUCNUM (bridge) is eliminated from the data.
#       i. 	IF a bridge element present in a bridge not meeting the criteria outlined in a. above (i.e. a bridge or STRUCNUM that is not eliminated from the data because that STRUCNUM IS present in all years under consideration) is not present and its condition state observed and recorded in all years being considered that bridge element (EN) is eliminated from the data.  (So eliminate the bridges first if they aren’t present in all the years then eliminate the elements from the bridges if the elements are not present in all the years for those bridges)
    
    
# Now begins the merging of different dataframes: Going to start by merging the two longest which are df2021CA and df2022CA
    
# The nomenclature for these merges is to place the two digit year corresponding to the next smallest df number at the beginning of the variable name as the merges are made, i.e. in the manner that df20_21_22 is merged below after the larger dfs of df2022CA and df2021CA have been merged.
    
    
#!!!
# Make the merging of the 7 dfs in one line starting below:
#df_merged = 
    
# df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)
    
#df =  pd.merge(df2022CA, df2021CA, df2020CA, df2019CA, df2018CA, df2017CA, df2016CA,  on=['STRUCNUM','EN'])
    
    
# !!!
#https://www.statology.org/pandas-merge-multiple-dataframes/
# look at link above to merge multiple dfs at once
    
    
# dfs = [df2016CA, df2017CA, df2018CA, df2019CA]
    
# df_final = reduce(lambda  left,right: pd.merge(left,right,on=['STRUCNUM'], how='outer'), dfs)
    
    
# df_final 46204 lines long.
    
# df191817 = df1918.merge(df_17, on = ['strucnum', 'EN'], how = 'left')
    
# merged_df = pd.merge(df1, df2, on='ID').merge(df3, on='ID').merge(df4, on='ID')
    
    
# Merge the dataframes for each year.
    
    
# MVPII: Make a dict of the df2016 thru df2022 dfs and just somehow say "merge all the dataframes in this dictionary...?  Possible?
    
    
    
# BEGIN Merge different dataframe years procedure - make the merge and possibly the creation of the df_all_yrs_merged variable merged dataframe,  and perhaps make the function doing the merge can start by first merging the largest df be merged with the next largest, the result of the first merge then being merged to the next largest remaining df, and so on iteratively until all dfs have been merged, the goal being to make the resulting df contain the largest possible number of bridges common among the group of dfs being merged.  
 
   
# Creates df_all_yrs_merged variable:

def merge_dataframes_by_length(dict_of_dfs, common_columns):
    # Sort dataframes by length in descending order
    sorted_dfs = sorted(dict_of_dfs.values(), key=len, reverse=True)
    
    # Start with the two longest dataframes
    merged_df = sorted_dfs[0].copy()
    
    for df in sorted_dfs[1:]:
        # Only consider the specified common columns
        common_cols = common_columns
    
        # Merge dataframes based on common columns
        merged_df = pd.merge(merged_df, df, on=common_cols, how='inner')
    # very important to indent the return statement below correctly!
    return merged_df


common_columns = ['STRUCNUM', 'EN']  # Specify the common columns
df_all_yrs_merged = merge_dataframes_by_length(filtered_df_nameToDF, common_columns)
print(df_all_yrs_merged)

# 01-29-2024 Make the additions/changes to the merge here- use the function above, call it again, create a new variable NOT called df_all_yrs_merged and call the function on the same dict of dataframes (filtered_df_nameToDF) but only call it on the single column STRUCNUM this time.  Then have the function call again and create the new dict of dataframes (with the aforementioned new variable name) and then filter all the different parts as was done previously- and make the new dict of dataframes the same as before, but with the extra bridge elements that will result, and the wherewithal to filter all the different EN's out, a larger set of data will result, then use this later to compare or make the difference of entries into a different variable and use to add to the dfs as needed to make the frequency between the observations into even numbers of hours or minutes without any digits to the right of the decimal to better run the ARIMA model.


# Begin getting lat and long coordinates procedure:

xlsx_path = 'CA15.xlsx'

# Read only the STRUCNUM 'Latitude' and 'Longitude' columns from the Excel file into a pandas DataFrame

loc_data = pd.read_excel(xlsx_path, usecols=['STRUCTURE_NUMBER_008', 'LAT_016', 'LONG_017'])

print(loc_data.dtypes)

# Need to make the latitude and longitude data useable, i.e. there are no decimals in the coordinates as they are given in the download from the FHWA.  
loc_data['LAT_016'] = loc_data['LAT_016'].astype(str)
# Place the decimal to the right of the second digit in the latitude column:
loc_data['LAT_016'] = loc_data['LAT_016'].apply(lambda x: x[:2] + '.' + x[2:])

loc_data['LONG_017'] = loc_data['LONG_017'].astype(str)

# longitude column, to right of third digit:
loc_data['LONG_017'] = loc_data['LONG_017'].apply(lambda x: x[:3] + '.' + x[3:])


# Convert types of lat and long data to numeric to allow for proper display on map
loc_data['LONG_017'] = pd.to_numeric(loc_data['LONG_017'], errors='coerce')

loc_data['LAT_016'] = pd.to_numeric(loc_data['LAT_016'], errors='coerce')

loc_data['LONG_017'] = loc_data['LONG_017'] * -1

# rename columns prior to merge of data into df_all_yrs_merged STRUCTURE_NUMBER_008 LAT_016 LONG_017 

cols_loc_data = {'STRUCTURE_NUMBER_008': 'STRUCNUM', 'LAT_016': 'LAT', 'LONG_017': 'LONG'}

loc_data.rename(columns=cols_loc_data, inplace=True)

# Check of data types
print(loc_data.dtypes)





"""
df16_17_18_19_20_21_22_23 = pd.merge(df2023CA, df2022CA, on=['STRUCNUM','EN'], suffixes=('_23', '_22')).merge(df2021CA, on=('STRUCNUM','EN'), suffixes=('_22', '_21')).merge(df2020CA, on=('STRUCNUM','EN'), suffixes=('_21', '_20')).merge(df2019CA, on=('STRUCNUM','EN'), suffixes=('_20', '_19')).merge(df2018CA, on=('STRUCNUM','EN'), suffixes=('_19', '_18')).merge(df2017CA, on=('STRUCNUM','EN'), suffixes=('_18', '_17')).merge(df2016CA, on=('STRUCNUM','EN'), suffixes=('_17', '_16'))
"""
    
# END Merge different dataframe years procedure
    
    

# MVP II: Merge on only STRUCNUM - get a larger dataset and use Python to replace missing data.  Was having difficulty with the size of the dataset when merging only on STRUCNUM.  

"""
df16_17_18_19_20_21_22 = pd.merge(df2022CA, df2021CA, on=['STRUCNUM'], suffixes=('_22', '_21')).merge(df2020CA,  on=['STRUCNUM'], suffixes=('_21', '_20')).merge(df2019CA, on=('STRUCNUM'), suffixes=('_20', '_19')).merge(df2018CA, on=('STRUCNUM'), suffixes=('_19', '_18')).merge(df2017CA, on=('STRUCNUM'), suffixes=('_18', '_17')).merge(df2016CA, on=('STRUCNUM'), suffixes=('_17', '_16'))
"""
    
# End merge of dataframes for each year.  
    
    
"""
df22_21 = df2022CA.merge(df2021CA, suffixes=['_22', '_21'], on=['STRUCNUM','EN'])
# pre merge the longest of the 2 dfs is 51033 lines long
# Post merge the length is 50104 lines long
# Switch to left merge: 51033
    
    
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
    
    
# The next longest is df2020CA at 50137 lines long
    
df20_22_21 = df22_21.merge(df2020CA, on=['STRUCNUM', 'EN'])
# pre merge the longest of the 2 dfs is df2021CA at 67758 lines long
# Post merge the length of df20_22_21 is 49754 lines long
    
    
# Change of suffixes post merge
    
#df.rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
    
df20_22_21.rename(columns={'FHWAED':'FHWAED_20', 'STATE':'STATE_20', 'EPN': 'EPN_20', 'TOTALQTY':'TOTALQTY_20', 'CS1':'CS1_20', 'CS2':'CS2_20', 'CS3':'CS3_20', 'CS4':'CS4_20', 'filename':'filename_20'}, inplace = True)
    
    
    
# Of the remaining dfs df2019CA is longest at 49671 lines long
    
df19_20_22_21 = df20_22_21.merge(df2019CA, on=['STRUCNUM', 'EN'])
    
# Post merge the length of df19_20_22_21 is 48734 lines long
    
# Change of suffixes post merge
    
df19_20_22_21.rename(columns={'FHWAED':'FHWAED_19', 'STATE':'STATE_19', 'EPN': 'EPN_19', 'TOTALQTY':'TOTALQTY_19', 'CS1':'CS1_19', 'CS2':'CS2_19', 'CS3':'CS3_19', 'CS4':'CS4_19','filename':'filename_19'}, inplace = True)
    
    
# Merge of df2018CA into the already merged years  2019 2020 2021 and 2022
    
df18_19_20_22_21 = df19_20_22_21.merge(df2018CA, on=['STRUCNUM', 'EN'])
    
# Post merge the length of df18_19_20_22_21 is 47978 lines long
    
# Change of suffixes post merge
    
df18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_18', 'STATE':'STATE_18', 'EPN': 'EPN_18', 'TOTALQTY':'TOTALQTY_18', 'CS1':'CS1_18', 'CS2':'CS2_18', 'CS3':'CS3_18', 'CS4':'CS4_18','filename':'filename_18'}, inplace = True)
    
# Merge 2017 into the df
df17_18_19_20_22_21 = pd.merge(df2017CA, df18_19_20_22_21, on=['STRUCNUM', 'EN'])
    
# Post merge the length of df17_18_19_20_22_21 is 46863 lines long
    
# Change of suffixes post merge
    
df17_18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_17', 'STATE':'STATE_17', 'EPN': 'EPN_17', 'TOTALQTY':'TOTALQTY_17', 'CS1':'CS1_17', 'CS2':'CS2_17', 'CS3':'CS3_17', 'CS4':'CS4_17','filename':'filename_17'}, inplace = True)
    
# Merge 2016 into the df
df16_17_18_19_20_22_21 = pd.merge(df2016CA, df17_18_19_20_22_21, on=['STRUCNUM', 'EN'])
    
# Post merge the length of df16_17_18_19_20_22_21 is 46204 lines long
    
# Change of suffixes post merge
    
    
    
df16_17_18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_16', 'STATE':'STATE_16', 'EPN': 'EPN_16', 'TOTALQTY':'TOTALQTY_16', 'CS1':'CS1_16', 'CS2':'CS2_16', 'CS3':'CS3_16', 'CS4':'CS4_16','filename':'filename_16'}, inplace = True)
"""
    
    
    
# Find out what happens to the EN column when the slices are made!! (06/15/2023)
    
# Original df16_17_18_19_20_22_21 length = 46204 lines long
# Use the expression below as a starting point for 11/27/2022
#abmt_rc = abmt_rc.loc[~((abmt_rc['CS2'] == 0.0) & (abmt_rc['CS1'] + abmt_rc['CS3'] + abmt_rc['CS4'] == 1.0)),:] 
# !!! 
# above expression is used to remove the rows from the dataframe that have a condition state within the set of observations being examined that equal zero- and as a result have the remainder of the CS on that row equal to 1 (i.e. checking that the sum of CS1-CS4 = the TOTALQTY - keep in mind that CS1-CS4 as they are used to compute regression are divided by the TOTALQTY and are manipulated as a fraction of the total)
    
# Ideas for removing the rows that don't have consistent TOTALQTY numbers for the years observed: 
# At least my latest is using the expression that the sum of TOTALQTY_17 thru TOTALQTY_22 divided by 7 must equal TOTALQTY_16.  This may not be that great of an idea because it's possible, although I would say unlikely, that the totals for each year could be right around the same number but not identical- but once added together and divided by 7 could equal the TOTALQTY_16 (i.e. the first year being observed in this case.) 
    
# Another idea is to say add all the TOTALQTY numbers on a row together and divide by 7 and make the TOTALQTY equal in each year by simply telling the computer to make it so.  I'm not a big fan of this idea because I feel it takes some of the usefulness of the observations away from us- when the TOTALs are different there is likely more to consider because the bridge may have been refurbished or widened or any number of things that represent the change in those TOTALQTY numbers year to year.  
# df_data[['CS1','CS2','CS3','CS4']] = df_data[['CS1','CS2','CS3','CS4']].div(df_data.TOTALQTY, axis=0)
    
# My attempt below.  Going to have to say make this feature part of MVP II.  
# df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21.loc[~((df16_17_18_19_20_22_21['TOTALQTY_16'] == df16_17_18_19_20_22_21[['TOTALQTY_16'+'TOTALQTY_17'+'TOTALQTY_18'+'TOTALQTY_19'+'TOTALQTY_20'+'TOTALQTY_16'+'TOTALQTY_21'+'TOTALQTY_22'].div(7, axis = 0)])),:]
    
    
# df_data[['CS1','CS2','CS3','CS4']] = df_data[['CS1','CS2','CS3','CS4']].div(df_data.TOTALQTY, axis=0)
    
#result.div(result.sum(axis=1), axis=0)
    
#abmt_rc = abmt_rc.loc[~((abmt_rc['CS2'] == 0.0) & (abmt_rc['CS1'] + abmt_rc['CS3'] + abmt_rc['CS4'] == 1.0)),:] 
    
    
    
#test = test[~((test['id'] == 1) & (test['num'] == 1))]
    
#dfd = df.drop(df[(df['1'] >= df['2'])].index)
    
#df.drop(df.index
#df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21[~df16_17_18_19_20_22_21.TOTALQTY_16.isin(['TOTALQTY_17', 'TOTALQTY_18', 'TOTALQTY_19', 'TOTALQTY_20', 'TOTALQTY_22', 'TOTALQTY_21'])]
#df = df.drop(df.index[df['colA']!=1.0])
    
# df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21.drop(df16_17_18_19_20_22_21.index[df16_17_18_19_20_22_21['TOTALQTY_16']!=1.0])
    
    
# wrong_tot = df16_17_18_19_20_22_21[ (df16_17_18_19_20_22_21['TOTALQTY_16'] != ['TOTALQTY_16']) | (['TOTALQTY_16'] != ['TOTALQTY_18']) | (['TOTALQTY_16'] != ['TOTALQTY_19']) | (['TOTALQTY_16'] != ['TOTALQTY_20']) | (['TOTALQTY_16'] != ['TOTALQTY_22']) | (['TOTALQTY_16'] == ['TOTALQTY_21'])].index
      
# df16_17_18_19_20_22_21[(['TOTALQTY_16'] == ['TOTALQTY_17']) & (['TOTALQTY_16'] == ['TOTALQTY_18']) & (['TOTALQTY_16'] == ['TOTALQTY_19']) & (['TOTALQTY_16'] == ['TOTALQTY_20']) & (['TOTALQTY_16'] == ['TOTALQTY_22']) & (['TOTALQTY_16'] == ['TOTALQTY_21'])].index
    
# drop these given row
# indexes from dataFrame
# df.drop(wrong_tot, inplace = True)
    
    
    
# df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21[~df16_17_18_19_20_22_21.TOTALQTY_16.isin(['TOTALQTY_17', 'TOTALQTY_18', 'TOTALQTY_19', 'TOTALQTY_20', 'TOTALQTY_22', 'TOTALQTY_21'])]
    
#index_names = df[ (df['Age'] >= 21) & (df['Age'] <= 23)].index
    
#df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21[(['TOTALQTY_16'] == ['TOTALQTY_17']) & (['TOTALQTY_16'] == ['TOTALQTY_18']) & (['TOTALQTY_16'] == ['TOTALQTY_19']) & (['TOTALQTY_16'] == ['TOTALQTY_20']) & (['TOTALQTY_16'] == ['TOTALQTY_22']) & (['TOTALQTY_16'] == ['TOTALQTY_21'])].index
    
# df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21[~df16_17_18_19_20_22_21.TOTALQTY_16.isin(['TOTALQTY_17', 'TOTALQTY_18', 'TOTALQTY_19', 'TOTALQTY_20', 'TOTALQTY_22', 'TOTALQTY_21'])]
    
# df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21.drop(df16_17_18_19_20_22_21.index[['TOTALQTY_16'] != (['TOTALQTY_17'])])
    
    
#df[~(df['Column B'].isin(df['Column A']) & (df['Column B'] != df['Column A']))]
    
#df17_18_19 = df17_18_19[~(df17_18_19['TOTALQTY_17'].isin(df17_18_19['TOTALQTY_18']) & (df17_18_19['TOTALQTY_19'] != df17_18_19['TOTALQTY_17']))]
    
#df17_18_19 = df17_18_19.drop(df17_18_19.index[df17_18_19['TOTALQTY_17'].isin(['TOTALQTY_18' != 'TOTALQTY_17', 'TOTALQTY' != 'TOTALQTY_17'])])
    
#df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21.drop(df16_17_18_19_20_22_21[(df16_17_18_19_20_22_21['TOTALQTY_16'] != df16_17_18_19_20_22_21['TOTALQTY_17'] != df16_17_18_19_20_22_21['TOTALQTY_18'] != df16_17_18_19_20_22_21['TOTALQTY_19'] != df16_17_18_19_20_22_21['TOTALQTY_20'] != df16_17_18_19_20_22_21['TOTALQTY_22'] != df16_17_18_19_20_22_21['TOTALQTY_21'])].index)
    
#df16_17_18_19_20_22_21 = df16_17_18_19_20_22_21.loc[~((df16_17_18_19_20_22_21['TOTALQTY_16'] != df16_17_18_19_20_22_21['TOTALQTY_17'] != df16_17_18_19_20_22_21['TOTALQTY_18'] != df16_17_18_19_20_22_21['TOTALQTY_19'] !=  df16_17_18_19_20_22_21['TOTALQTY_20'] != df16_17_18_19_20_22_21['TOTALQTY_21'] != df16_17_18_19_20_22_21['TOTALQTY_22'])),:]
                                                          
    
    
# NEED TO CONVERT THE CS1 thru CS4s TO PERCENTAGES AS I DID BEFORE!!!!
""" Perhaps do that part after I've made all the new dfs.  """
    
    
# df16, df17, df18, df19, df20, df21 & df22 represent the subsets of df16_17_18_19_20_22_21 that will be removed from that dataframe to make individual dataframes to be concatenated later.  
    
    
# Select columns of the dataframe df16_17_18_19_20_22_21 to make the dataframe for each individual year:
        
# These datafames will be in the general form of the following column headings: filename | STRUCNUM | EN | TOTALQTY | CS1 | CS2 | CS3 | CS4 and will be specific to the year represented in the variable name dfXX where XX is the 2 digit year (in this case ranging from 17 to 21).
    
# !!!
# Not sure 
    
"""
# for year 2016
df16 = df16_17_18_19_20_22_21.iloc[:,[0,3,4,6,7,8,9,10]]
    
# for year 2017
df17 = df16_17_18_19_20_22_21.iloc[:,[11,3,4,15,16,17,18,19]]
    
# for year 2018
df18 = df16_17_18_19_20_22_21.iloc[:,[20,3,4,24,25,26,27,28]]
    
# for year 2019
df19 = df16_17_18_19_20_22_21.iloc[:,[29,3,4,33,34,35,36,37]]
    
# for year 2020
df20 = df16_17_18_19_20_22_21.iloc[:,[38,3,4,42,43,44,45,46]]
    
# !!!
    
# for year 2022
df22 = df16_17_18_19_20_22_21.iloc[:,[47,3,4,51,52,53,54,55]]
    
# for year 2021
df21 = df16_17_18_19_20_22_21.iloc[:,[56,3,4,60,61,62,63,64]]
"""
    
    
    
# for year 2016
#df16 = df16_17_18_19_20_22_21.iloc[:,[2,3,5,6,7,8,9,10]]
    
# for year 2017
#df17 = df16_17_18_19_20_22_21.iloc[:,[11,3,4,15,16,17,18,19]]
    
# for year 2018
#df18 = df16_17_18_19_20_22_21.iloc[:,[20,3,4,24,25,26,27,28]]
    
# for year 2019
#df19 = df16_17_18_19_20_22_21.iloc[:,[29,3,4,33,34,35,36,37]]
    
# for year 2020
#df20 = df16_17_18_19_20_22_21.iloc[:,[38,3,4,42,43,44,45,46]]
    
# !!!
    
# for year 2022
#df22 = df16_17_18_19_20_22_21.iloc[:,[47,3,4,51,52,53,54,55]]
    
# for year 2021
#df21 = df16_17_18_19_20_22_21.iloc[:,[56,3,4,60,61,62,63,64]]
    
    
# Begin dataframe slicing of non-common columns procedure
df_all_yrs_merged
    
    
# 12.31.2023 stopped here for the night
    
    
# identify the columns without suffixes (i.e. no '_')
common_cols = [col for col in df_all_yrs_merged.columns if '_' not in col]
    
# group columns by their suffixes
group_cols = {suffix: df_all_yrs_merged.filter(like=suffix).columns for suffix in set(col.split('_')[-1] for col in df_all_yrs_merged.columns if '_' in col)}

# slice out the dataframes based on the suffix that refers to year.  
sliced_dfs = {}
for suffix, group_col in group_cols.items():
    sliced_dfs[suffix] = df_all_yrs_merged.loc[:, common_cols + group_col.tolist()]

# Begin remove all suffixes (i.e. the "_" and anything to the right of it) from all dataframe headings in dictionary

    
def rmv_suffix_fr_dict_dfs_cols(dict_of_dfs):
    for df_name, df in dict_of_dfs.items():
        # Iterate through columns in the DataFrame
        for col in df.columns:
            # Remove the suffix (i.e., everything to the right of and including "_")
            new_col_name = col.split('_')[0]
            # Rename the column in the DataFrame
            df.rename(columns={col: new_col_name}, inplace=True)
    
if __name__ == "__main__":
    
# function call
    rmv_suffix_fr_dict_dfs_cols(sliced_dfs)


# End remove all suffixes (i.e. the "_" and anything to the right of it) from all dataframe headings in dictionary



# Sort the dictionary keys numerically
sorted_keys = sorted(sliced_dfs.keys())

df_data = pd.concat([sliced_dfs[key] for key in sorted_keys])

    
# 12.31.2023 stopped here for the night
    
# End dataframe slicing of non-common columns procedure df_all_yrs_concat
    
"""
# Begin slicing of df_16_17_18_19_20_21_22_23 dataframe
# How could this be automated>? The slicing.  
    
df16 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,68,69,70,71,72,73]]
    
df17 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,59,60,61,62,63,64]]
    
df18 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,50,51,52,53,54,55]]
    
df19 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,41,42,43,44,45,46]]
    
df20 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,32,33,34,35,36,37]]
    
df21 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,23,24,25,26,27,28]]
    
df22 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,14,15,16,17,18,19]]
    
df23 = df16_17_18_19_20_21_22_23.iloc[:,[2,3,5,6,7,8,9,10]]
    
# End slicing of df_16_17_18_19_20_21_22_23 dataframe
"""
    
# for year 2018
#df18 = df18_17_21_20_19.iloc[:,[0,3,4,6,7,8,9,10]]
    
# for year 2017
#df17 = df18_17_21_20_19.iloc[:,[11,3,4,15,16,17,18,19]]
    
# for year 2021
#df21 = df18_17_21_20_19.iloc[:,[20,3,4,24,25,26,27,28]]
    
# for year 2020
#df20 = df18_17_21_20_19.iloc[:,[29,3,4,33,34,35,36,37]]
    
# for year 2019
#df19 = df18_17_21_20_19.iloc[:,[38,3,4,42,43,44,45,46]]
    
    
# BEGIN Rename columns
"""
df16_17_18_19_20_21_22_23.rename(columns={'FHWAED':'FHWAED_16', 'STATE':'STATE_16', 'EPN': 'EPN_16', 'TOTALQTY':'TOTALQTY_16', 'CS1':'CS1_16', 'CS2':'CS2_16', 'CS3':'CS3_16', 'CS4':'CS4_16','filename':'filename_16'}, inplace = True)
"""
# END Rename columns
    
    
# Change the column headings of the first df- to avoid any confusion of the different suffixes associated with the different years:
    
        
# Change somehow to a df1st instead of df16 because this could mean avoiding hard coding something there.  
"""
df16.columns = ['STRUCNUM', 'EN', 'TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4',  'filename']
"""
    
# Begin create df_data dataframe
# concatenate the dataframes to one another starting with year 2016
    
# The dataframe holding all the years in order will be called df_data
"""
df_data =pd.DataFrame(np.concatenate([df16.values, df17.values, df18.values, df19.values, df20.values, df21.values, df22.values, df23.values], axis=0), columns=df16.columns)
"""
# End create df_data dataframe
    
# !!!
    
# Convert the columns to numeric - admittedly the filename probably does not need to be numeric, but I'm trying to get this done.
    
df_data[['TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']] = df_data[['TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']].apply(pd.to_numeric, errors='coerce')
    
    
# Divide the CS1 thru CS4 by TOTALQTY to make the condition state into a percentage of the total element per bridge
    
df_data[['CS1','CS2','CS3','CS4']] = df_data[['CS1','CS2','CS3','CS4']].div(df_data.TOTALQTY, axis=0)
    
    
    
# el_names means Element Names
# https://stackoverflow.com/questions/39502079/use-strings-in-a-list-as-variable-names-in-python
        
    
# !!!
    
# Use the key of a dictionary as the criteria to select rows from a dataframe and set the resultant rows equal to the value corresponding to the key
    
    
""" 06/07/22 """
    
    
    
# Filter the rows out of the df_data dataframe based on the largest possible data set where each STRUCNUM has all the EN attached to it observed over all years being considered- i.e. the set of EN attached to a particular STRUCNUM is the set of EN not necessarily attached to or observed in one particular year- but the most EN observed over all the years of data being considered (eliminating repeats).
    
# Begin filter individual dataframes from df_data (concatenated overall dataframe) procedure: 
    
# Get unique element numbers (el_numbers) from the 'EN' column
    
el_numbers = df_data['EN'].unique()
    
# The EN present in the resulting dataframe (for the state of California) are 64 of the possible 124 that are recognized by the FHWA.
    
    
    
# Create an empty dictionary to store the smaller dataframes
    
element_dfs = {}
    
    
# Iterate over the element numbers and create smaller dataframes filtered out based on the individual element numbers
    
for el_number in el_numbers:
    # Use getattr to dynamically create a dataframe for each group
        
    element_df = getattr(df_data.loc[df_data['EN'] == el_number], 'copy')()    
    element_dfs[el_number] = element_df
    
    
# End filter individual dataframes from df_data (concatenated overall dataframe) procedure: 
    
    
    
    
# !!! 
    
# 06/17/2023: To line of code above: element_dfs[el_number] = element_df, the code is working i.e. the dfs are parsed in correctly, the merge of the respective years of the dataframes are occurring correctly, 
    
    
# df_nameToDF is the dictionary of dataframes as created when the df_names are matched up with the data corresponding to the xml file from which the data is read at parse.  
    
    
    
el_names = {'12': 'deck_rc',
            '13': 'deck_pc',
       	    '15': 'topFlg_pc', 
            '16': 'topFlg_rc',
   	        '28': 'stDeck_og', 
            '29': 'stDeck_cfg',
            '30': 'stDeck_corrOrtho',
            '31': 'deck_timb',
            '38': 'slab_rc',
            '39': 'slab_pc',
            '54': 'slab_timb',
            '60': 'deck_other',
            '65': 'slab_other',
            '102': 'cwBg_steel',
            '104': 'cwBg_pc',
            '105': 'cwBg_rc',
            '106': 'cwBg_other',
            '107': 'oGb_steel',
            '109': 'oGb_pc',
            '110': 'oGb_rc',
            '111': 'oGb_timb',
            '112': 'oGb_other',
            '113': 'stringer_steel',
            '115': 'stringer_pc',
            '116': 'stringer_rc',
            '117': 'stringer_timb',
            '118': 'stringer_other',
            '120': 'truss_steel',
            '135': 'truss_timb',
            '136': 'truss_other',
            '141': 'arch_steel',
            '142': 'arch_other',
            '143': 'arch_pc',
            '144': 'arch_rc',
            '145': 'arch_masonry',
            '146': 'arch_timb',
            '147': 'cbl_mSt',
            '148': 'cbl_secSt',
            '149': 'cbl_secOthr',
            '152': 'flrB_steel',
            '154': 'flrB_pc',
            '155': 'flrB_rc',
            '156': 'flrB_timb',
            '157': 'flrB_other',
            '161': 'spph',
            '162': 'sgp',
            '170': 'rrcf',
            '171': 'miscSS',
            '180': 'eqrcII',
            '181': 'eqrcC1',
            '182': 'eqrc_Othr',
            '202': 'col_st',
            '203': 'col_othr',
            '204': 'col_pc',
            '205': 'col_rc',
            '206': 'col_timb',
            '207': 'twr_st',
            '208': 'tres_timb',
            '210': 'pw_rc',
            '211': 'pw_othr',
            '212': 'pw_timb',
            '213': 'pw_mas',
            '215': 'abmt_rc',
            '216': 'abmt_timb',
            '217': 'abmt_mas',
            '218': 'abmt_othr',
            '219': 'abmt_steel',
            '220': 'pcf_rc',
            '225': 'pile_st',
            '226': 'pile_pc',
            '227': 'pile_rc',
            '228': 'pile_timb',
            '229': 'pile_othr',
            '231': 'pc_steel',
            '233': 'pc_PrConc',
            '234': 'pc_rc',
            '235': 'pc_timb',
            '236': 'pc_othr',
            '240': 'culv_st',
            '241': 'culv_rc',
            '242': 'culv_timb',
            '243': 'culv_othr',
            '244': 'culv_mas',
            '245': 'culv_pc',
            '250': 'tunnel',
            '251': 'pile_castSh',
            '252': 'pile_castDr',
            '254': 'cSh_stFH',
            '255': 'cSh_stPH',
            '256': 'slopeScP',
            '300': 'joint_sse',
            '301': 'joint_ps',
            '302': 'joint_cs',
            '303': 'joint_aws',
            '304': 'joint_oe',
            '305': 'joint_awo',
            '306': 'joint_othr',
            '307': 'joint_ap',
            '308': 'joint_ssp',
            '309': 'joint_sf',
            '310': 'brg_el',
            '311': 'brg_mov',
            '312': 'brg_ec',
            '313': 'brg_fxd',
            '314': 'brg_pot',
            '315': 'brg_dsk',
            '316': 'brg_othr',
            '320': 'appSl_pc',
            '321': 'appSl_rc',
            '330': 'br_m',
            '331': 'br_rc',
            '332': 'br_timb',
            '333': 'br_othr',
            '334': 'br_mas',
            '510': 'dws_ac',
            '511': 'dws_cp',
            '512': 'dws_ep',
            '513': 'dws_timb',
            '515': 'spc_p',
            '516': 'spc_galv',
            '517': 'spc_ws',
            '520': 'rsps',
            '521': 'cpc',
            '522': 'deck_memb'}
    
    
"""
# Deck and slabs, 13 elements
    
deck_rc = getattr(element_df, '12', None)
    
deck_pc = getattr(element_df, '13', None)
    
topFlg_pc = getattr(element_df, '15', None)

topFlg_rc = getattr(element_df, '16', None)
# 43050 rows long

stDeck_og = getattr(element_df, '28', None)

stDeck_cfg = getattr(element_df, '29', None)

stDeck_corrOrtho = getattr(element_df, '30', None)

deck_timb = getattr(element_df, '31', None)

slab_rc = getattr(element_df, '38', None)

slab_pc = getattr(element_df, '39', None) # None in the data, MVP II

slab_timb = getattr(element_df, '54', None)

deck_other = getattr(element_df, '60', None) # None in the data, MVP II
    
slab_other = getattr(element_df, '65', None) # None in the data, MVP II

    # End deck and slabs

    # Superstructure, 38 elements
    
cwBg_steel = getattr(element_df, '102', None)

cwBg_pc = getattr(element_df, '103', None) # None in the data, MVP II

cwBg_rc = getattr(element_df, '105', None)

cwBg_other = getattr(element_df, '106', None) # None in the data, MVP II

oGb_steel = getattr(element_df, '107', None)

oGb_pc = getattr(element_df, '109', None)

oGb_rc = getattr(element_df, '110', None)

oGb_timb = getattr(element_df, '111', None)

oGb_other = getattr(element_df, '112', None) # None in the data, MVP II

stringer_steel = getattr(element_df, '113', None)

stringer_pc = getattr(element_df, '115', None)

stringer_rc = getattr(element_df, '116', None)

stringer_timb = getattr(element_df, '117', None)

stringer_other = getattr(element_df, '118', None) # None in the data, MVP II

truss_steel = getattr(element_df, '120', None)

truss_timb = getattr(element_df, '135', None) # None in the data, MVP II

truss_other = getattr(element_df, '136', None) # None in the data, MVP II

arch_steel = getattr(element_df, '141', None)

arch_other = getattr(element_df, '142', None) # None in the data, MVP II

arch_pc = getattr(element_df, '143', None)

arch_rc = getattr(element_df, '144', None)

arch_masonry = getattr(element_df, '145', None) # None in the data, MVP II

arch_timb = getattr(element_df, '146', None) # None in the data, MVP II

cbl_mSt = getattr(element_df, '147', None)

cbl_secSt = getattr(element_df, '148', None) # None in the data, MVP II

cbl_secOthr = getattr(element_df, '149', None) # None in the data, MVP II

flrB_steel = getattr(element_df, '152', None)

flrB_pc = getattr(element_df, '154', None)

flrB_rc = getattr(element_df, '155', None)

flrB_timb = getattr(element_df, '156', None)

flrB_other = getattr(element_df, '157', None) # None in the data, MVP II

spph = getattr(element_df, '161', None) # None in the data, MVP II

sgp = getattr(element_df, '162', None)

rrcf = getattr(element_df, '170', None) # None in the data, MVP II

miscSS = getattr(element_df, '171', None) # None in the data, MVP II

eqrcII = getattr(element_df, '180', None) # None in the data, MVP II

eqrcC1 = getattr(element_df, '181', None) # None in the data, MVP II

eqrc_Othr = getattr(element_df, '182', None) # None in the data, MVP II

    # End Superstructure
    
    # Substructure, 40 elements

col_st = getattr(element_df, '202', None)

col_othr = getattr(element_df, '203', None) # None in the data, MVP II

col_pc = getattr(element_df, '204', None)

col_rc = getattr(element_df, '205', None)
# 34405 rows long

col_timb = getattr(element_df, '206', None)

twr_st = getattr(element_df, '207', None)

tres_timb = getattr(element_df, '208', None) # None in the data, MVP II

pw_rc = getattr(element_df, '210', None)

pw_othr = getattr(element_df, '211', None) # None in the data, MVP II

pw_timb = getattr(element_df, '212', None)

pw_mas = getattr(element_df, '213', None)

abmt_rc = getattr(element_df, '215', None)
# 61376 rows long

abmt_timb = getattr(element_df, '216', None)

abmt_mas = getattr(element_df, '217', None)

abmt_othr = getattr(element_df, '218', None)

abmt_steel = getattr(element_df, '219', None)

pcf_rc = getattr(element_df, '220', None)

pile_st = getattr(element_df, '225', None)

pile_pc = getattr(element_df, '226', None)

pile_rc = getattr(element_df, '227', None)

pile_timb = getattr(element_df, '228', None)

pile_othr = getattr(element_df, '229', None)

pc_steel = getattr(element_df, '231', None)

pc_PrConc = getattr(element_df, '233', None)

pc_rc = getattr(element_df, '234', None)

pc_timb = getattr(element_df, '235', None)

pc_othr = getattr(element_df, '236', None) # None in the data, MVP II

culv_st = getattr(element_df, '240', None)

culv_rc = getattr(element_df, '241', None)

culv_timb = getattr(element_df, '242', None) # None in the data, MVP II

culv_othr = getattr(element_df, '243', None)

culv_mas = getattr(element_df, '244', None)

culv_pc = getattr(element_df, '245', None)

tunnel = getattr(element_df, '250', None) # None in the data, MVP II

pile_castSh = getattr(element_df, '251', None) # None in the data, MVP II

pile_castDr = getattr(element_df, '252', None) # None in the data, MVP II

cSh_stFH = getattr(element_df, '254', None) # None in the data, MVP II

cSh_stPH = getattr(element_df, '255', None) # None in the data, MVP II

slopeScP = getattr(element_df, '256', None) # None in the data, MVP II

    # End Substructure
    
    # Joints, 10 elements

joint_sse = getattr(element_df, '300', None)

joint_ps = getattr(element_df, '301', None)
# 23128 rows long

joint_cs = getattr(element_df, '302', None)

joint_aws = getattr(element_df, '303', None)

joint_oe = getattr(element_df, '304', None)

joint_awo = getattr(element_df, '305', None)

joint_othr = getattr(element_df, '306', None)

joint_ap = getattr(element_df, '307', None) # None in the data, MVP II

joint_ssp = getattr(element_df, '308', None) # None in the data, MVP II

joint_sf = getattr(element_df, '309', None) # None in the data, MVP II

    # End Joints
    
    # Bearings, 7 elements

brg_el = getattr(element_df, '310', None)

brg_mov = getattr(element_df, '311', None)

brg_ec = getattr(element_df, '312', None)

brg_fxd = getattr(element_df, '313', None)

brg_pot = getattr(element_df, '314', None)

brg_dsk = getattr(element_df, '315', None)

brg_othr = getattr(element_df, '316', None)

    # End Bearings
    
    # Approach Slabs, 2 elements

appSl_pc = getattr(element_df, '320', None) # None in the data, MVP II

appSl_rc = getattr(element_df, '321', None) # None in the data, MVP II

    # End Approach Slabs
    
    # Railings, 5 elements
    
br_m = getattr(element_df, '330', None)

br_rc = getattr(element_df, '331', None)

br_timb = getattr(element_df, '332', None)

br_othr = getattr(element_df, '333', None)

br_mas = getattr(element_df, '334', None)

    # End Railings
    
    # Wearing Surfaces, 10 elements
    
dws_ac = getattr(element_df, '510', None)

dws_cp = getattr(element_df, '511', None) # None in the data, MVP II

dws_ep = getattr(element_df, '512', None) # None in the data, MVP II

dws_timb = getattr(element_df, '513', None) # None in the data, MVP II

spc_p = getattr(element_df, '515', None)

spc_galv = getattr(element_df, '516', None) # None in the data, MVP II

spc_ws = getattr(element_df, '517', None) # None in the data, MVP II

rsps = getattr(element_df, '520', None) # None in the data, MVP II

cpc = getattr(element_df, '521', None) # None in the data, MVP II

deck_memb = getattr(element_df, '522', None) # None in the data, MVP II

    # End Wearing Surfaces
    
    # End Elements

"""

# Rationale for the replacement of data for deck_rc is that the subset of data will consist of all bridges that have observations in all years AND at least one EN observation in one year- thus replacing the the EN observations for years where no data is present but at least one observation is present in at least one year for a bridge.      

# 62 of the possible EN are NoneType objects (i.e. there are no observations of those EN present across all years being considered) which is not surprising because the list of total possible elements is exhaustive and many of the total of  124 elements are specialized types of construction that do not see use in most typical highway briidges.  


# Create time column for the plots.  
# Make the required dfs and then make plots and perform regression.  
# Create a column to compute the percentage of each CS (condition state) as means to determine when that CS might overtake all of the elements to which it pertains.  


# I expect the most commonly used elements present in the data to be the type with the suffix _rc meaning reinforced concrete and _pc meaning prestressed concrete.  

# !!!
# simple linear regression for deck_rc or reinforced concrete deck
# !!!



sns.set(style="whitegrid", color_codes=True)

# https://stackoverflow.com/questions/70949098/how-to-work-around-the-date-range-limit-in-pandas-for-plotting

# abmt_rc Bridge abutments, reinforced concrete

# Going to do some work to this dataframe before using it to make a regression model

# The abmt_rc df is 61376 lines long and has 7 years of data 61376/7 = 8768 entries per year.  


# Create a datetime object to use for the abmt_rc df:
# Note how the range is stopped at 12/30/2021 to make the number of intervals between observations the same for each year and to allow for the 2020 leap year.  00:00:00.000000

# iloc[row slicing, column slicing]

# slicing the deck_rc dataframe into its individual years to assign date_range accurately.
  
# 8768 periods/year

# Let's remove any CS1 entries in the abmt_rc dataframe that have progressed through CS1 (i.e. meaning CS1 = 0) entirely and will only produce subsequent CS1 observations if an outside influence acts upon that bridge element.   I take outside influence to generally mean replacement of the element completely which would mean the element was new and by default in a condition state of CS1- although there may be refurbishment or repair actions that may return the element or a portion of that element to a state of CS1.  

# So originally after the abmt_rc entries were placed into their own dataframe the number of rows was 61376.  

# !!!
# Here is where the abmt_rc df is important to look at, rather than create more variables and make things hard to follow I'm going to adjust the equation below so that the CS under review is different while looking for a line of best fit in the different condition states.  

# !!!

# Having gotten the dictionary of dataframes known as element_dfs to hold the dataframes of each individual bridge element found in the data, I need to make the dataframes into a form that will allow for creation of data visualization.


# Can a dictionary of dataframes have its keys changed by using another dictionary the keys of which match the original dataframe keys and join the keys in the original dataframe to the values in the second dictionary? 



# Recall that you want to make the values of the new_keys_dict come first in the join- with the original_dict keys following the underscore in the resulting original_dict



# Place the value (description) from the el_names dictionary at the front of the existing element_dfs keys to make more acceptable variable names.  In other words the description e.g. topFlg_rc whose corresponding element number is 16 would take the form topFlg_rc_16 after this next function is used.  

# Generic dict variable name: new_dict_ind_bdg_elems (newly created dictionary of individual bridge elements) i.e. the dictionary of  dataframes with the specific bridge elements separated out of the larger merge that created df_data.

# Generic dict variable name: exist_dict_mod_vals i.e. the existing dictionary,  values of which are to be modified- in this case adding a suffix.

def add_key_desc(element_dfs, el_names):
    # Get a list of the original keys
    original_keys = list(element_dfs.keys())

    # Iterate over the original keys
    for key in original_keys:
        # Check if the key needs to be changed
        if key in el_names:
            new_key = '_'.join([el_names[key], key.replace(" ", "")])
            element_dfs[new_key] = element_dfs.pop(key)

    return element_dfs

add_key_desc(element_dfs, el_names)

# End add_key_desc putting value from el_names in front of keys of element_dfs dictionary


# I plan to use the CS1 condition state data as the data I will investigate for now.  I feel it necessary to point this out because the functions I plan to use may tend to remove the rows of data from the dataframe when they are not applicable such as in the case of outliers or as I have been doing below removing 1s and zeros.  

# And just as I say that it occurs to me that Perhaps pulling out each CS case and evaluating them individually may be a good idea, or rather to at least create the functions to do so- rather than just making additional dicts of dataframes that would have the CS2 thru CS4 data still inside the dataframe when that data will most likely have had rows pulled out that were not outliers or 1s or zeros thus having them removed would not aid in the well fitted-ness of the model, so I may go to a practice where a dict of dataframes is created that pulls out all of the unnecessary columns, and only the CS column being considered will be left in.  

# How many zeros are there the condition state CS1 thru CS4 columns in each dataframe ?

def how_many_zeros(zeros_dict, col_names):
    zero_occurrences = {}
    
    for key, dataframe in zeros_dict.items():
        no_rows = len(dataframe)
        zeros_count = []
        
        for col_name in col_names:
            if col_name in dataframe.columns:
                zeros_per_col = dataframe[col_name].value_counts()
                zeros_found_col = zeros_per_col.get(0, 0)
                percentage_zeros = (zeros_found_col / len(dataframe[col_name])) * 100
                zeros_count.append(f"{col_name}: {zeros_found_col} ({percentage_zeros:.2f}%)")
        
        zero_occurrences[key] = [no_rows] + zeros_count
    
    return zero_occurrences
zeros_dict_dfs = element_dfs.copy()
col_names = ['CS1', 'CS2', 'CS3', 'CS4']

zero_occurrences = how_many_zeros(zeros_dict_dfs, col_names)

for key, counts in zero_occurrences.items():
    print(f"Number of zeros in dataframe '{key}': {counts}")

# End how_many_zeros


# How many ones are there in the condition state CS1 thru CS4 columns in each dataframe ?


def how_many_ones(ones_dict, col_names): # ones_dict refers to the dictionary that is examined for the number and percentage of 1s in its CS1-CS4 columns.  
    one_occurrences = {}

    for key, dataframe in ones_dict.items():
        no_rows = len(dataframe)
        ones_count = []
        
        for col_name in col_names:
            if col_name in dataframe.columns:
                ones_per_col = dataframe[col_name].value_counts()
                ones_found_col = ones_per_col.get(1, 0)
                percentage_ones = (ones_found_col / len(dataframe[col_name])) * 100
                ones_count.append(f"{col_name}: {ones_found_col} ({percentage_ones:.2f}%)") # 2 decimal places
            
        one_occurrences[key] = [no_rows] + ones_count
                
    return one_occurrences


ones_dict_dfs = element_dfs.copy()
col_names = ['CS1', 'CS2', 'CS3', 'CS4']

one_occurrences = how_many_ones(ones_dict_dfs, col_names)

for key, counts in one_occurrences.items():
    print(f"Number of ones in dataframe '{key}': {counts}")

# End how_many_ones


# Remove 1s (and zeros (?)) from the dataframes within the dicts and save the dicts to a new variable that will point out the condition state that will be examined using the data from that dictionary


# Is there a way to slice off one column and then copy another and attach it to the slice?
# Slice the dataframes to allow the original data in the dataframe when first pulled from the element_dfs dictionary to stay the same except for the column sliced off - make the data from a single CS category into a numpy array and make the (03-30-2024, just noticed that I trailed off here and didn't finish the thought...)

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



# !!!
# Move the import below later!
from model1 import rem_cols_mk_df_model1



# Begin remove columns from element_dfs except for CS1

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




# Begin Visually check the outliers in the dataframes of the dict
# Function to plot boxplots for each DataFrame in the dictionary known as element_dfs_CS1

# boxplot cannot take a dictionary as a direct argument

"""
def boxplots_CS1(element_dfs_CS1):
    for df_keyname, df in element_dfs_CS1.items():
        
        # Create a boxplot for each DataFrame
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=df)
                plt.title(f"Boxplot for {df_keyname}")
                plt.show()

# Call the function to plot boxplots
boxplots_CS1(element_dfs_CS1)
"""
# End Visually check the outliers in the dataframes of the dict


# To examine the regression of the elements that are in CS1 remove the 1s from the dictionary 

# Remove the rows with 1s (ones) in them from the dataframes based on the presence of 1s in the CS1 column


# element_dfs_CS1 = {} # new_data
# person = 'Mike'
# references
    # name: person --> 44
# memory
    # address: 44 --> value --> Mike
# Reference means the name of the variable, and the 
# The reference means the variable, and the reference has a name (person in this case) AND the reference creates a memory address (or at least the value for the memory address) The memory address actually holds the value of the variable corresponding to the name of the reference. 

# So the name given to a variable is like the shingle for an office, and the reference with the corresponding name of the variable is like your address book: There's the name of what lives at the address (could be a person like in this case- so one would say "we know a person lives at this address-"), followed by the number of the address (i.e. where the memory corresponding to the value of the variable is located).  

# The memory for the name of the variable holds the value of the variable, the value at that memory location represents the specifics of the variable at the (i.e. a name of a person in this case, someone named Mike) and the address created in reference represents Mike's address, so specifically Mike has an address the number of which is created in reference when the variable is created.  

# The object in memory is the actual location of the address that holds the value of a variable, and person is a reference.

# Python has names that refer to objects.  Objects exist separately from names and names exist separately from the objects to which they refer.  (Thusly the address book analogy, the name being the shingle, the reference to the address being the address book, and the object being the )




#def change_string(input_string:str) -> None:
""" Notice that this functions doesn't return anything! """
 #   input_string += 'a'

#def change_list(input_list:list) -> None:
""" Notice that this functions doesn't return anything! """
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



# !!!
# Begin eliminate outliers procedure

# !!!
# Fixed below
# 08/19/2023: The outliers procedure is not removing the rows from the dataframes- this needs to be either changed or addressed through data replacement 

def elim_outliers(dict_of_dfs, column_name, z_threshold=2):
    dict_minus_outls = {}

    for key, dataframe in dict_of_dfs.items():
        # Calculate z-scores for the column
        z_scores = (dataframe[column_name] - dataframe[column_name].mean()) / dataframe[column_name].std()
        
        # Create a mask to identify outlier rows
        mask = abs(z_scores) <= z_threshold
        
        # Apply the mask to remove outlier rows
        dict_minus_outls[key] = dataframe[mask]
    
    return dict_minus_outls

# Which column is being evaluated?
column_name = 'CS1'

# Call the function to remove outliers
outls_rmvd_CS1 = elim_outliers(element_dfs_CS1, column_name)

# End eliminate outliers procedure







"""
# Function to plot boxplots for each DataFrame in the dictionary
def plot_boxplots_for_dict_of_dfs(element_dfs_CS1):
    for df_name, df in element_dfs_CS1.items():
        # Create a boxplot for each DataFrame
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df)
        plt.title(f"Boxplot for {df_name}")
        plt.show()

# Call the function to plot boxplots
plot_boxplots_for_dict_of_dfs(element_dfs_CS1)

"""


# Need to make dict_dicts_CS1 which is to consist of the following existing dictionaries (a dictionary based on which of the CS columns is being examined):
    #element_dfs_CS1, (dfs)
    #element_dfs_CS1_0s, (dfs)
    #element_dfs_CS1_1s, (dfs)
    #outls_rmvd_CS1 (dfs)



# initialize empty dictionary for the dictionary of dictionaries of all the dictionaries listed above:
dict_dicts_CS1 = {}

# Make the existing dictionaries into a list:
dicts_CS1_list = [element_dfs_CS1, element_dfs_CS1_0s, element_dfs_CS1_1s, outls_rmvd_CS1]

# because dictionary objects are not hashable, they cannot be used as keys in another dictionary.  
dict_names_CS1 = ['element_dfs_CS1', 'element_dfs_CS1_0s', 'element_dfs_CS1_1s', 'outls_rmvd_CS1']

dict_dicts_CS1 = {var_name: globals()[var_name] for var_name in dict_names_CS1}

# dict_of_dicts = {var_name: locals()[var_name] for var_name in dict_names}

"""
# sample_module.py

def create_dict_of_dicts(*dicts):
    return {var_name: data_dict for var_name, data_dict in zip(dict_names, dicts)}

# sample main.py file

import sample_module

# Sample separate dictionaries
dict1 = {'key1': 'df1', 'key2': 'df2', ...}  # Replace 'df1', 'df2' with your actual DataFrames
dict2 = {'key1': 'df3', 'key2': 'df4', ...}  # Replace 'df3', 'df4' with your actual DataFrames
# Add more dictionaries as needed

# List of existing dictionaries
list_of_original_dicts = [dict1, dict2]  # Add more dictionaries as needed

# List of variable names
dict_names = ['dict1', 'dict2']  # Match the order of variable names to dictionaries

# Use the function from sample_module to create dict_of_dicts
dict_of_dicts = sample_module.create_dict_of_dicts(*list_of_original_dicts)

# Now dict_of_dicts contains the desired structure with variable names as keys
"""




# Begin year extract procedure
# Use the numeric portion of the entries in the filename columns of the dataframes to use as a precursor for the year during which each observation of the condition states of each bridge occurred.  

# !!!  08-22-2023

# df
# Get the year associated with each observation from the filename in the row of that observation.  


# BEGIN extract year from filename procedure

def getyr_fr_filename(dict_of_dicts):
    new_dict_of_dicts = {}
    
    for key, dictionary in dict_of_dicts.items():
        new_dictionary = {}
        
        for sub_key, dataframe in dictionary.items():
            new_dataframe = dataframe.copy()  # Create a new DataFrame
            
            # Extract digits from filename using regex
            digits_in_filename = new_dataframe['filename'].str.extract(r'(\d{4})')
            
            # Convert digits to int data type
            new_dataframe['year'] = digits_in_filename.astype(int)
            
            new_dictionary[sub_key] = new_dataframe
        
        new_dict_of_dicts[key] = new_dictionary
    
    return new_dict_of_dicts

# Dictionary of dictionaries to examine is dict_dicts_CS1


# Apply the function to extract digits and create 'year' column
new_dict_dicts_CS1 = getyr_fr_filename(dict_dicts_CS1)

# END extract year from filename procedure


# End year extract procedure
# Carry on with *Spread Time in DataFrame in order to create the time column in a dictionary of dictionaries- the variable of which is call new_dict_dicts_CS1

"""
Plots to make:
    element_dfs_CS1 Not. Sure.
    element_dfs_CS1_0s
    element_dfs_CS1_1s
    element_dfs_CS1 (but after being run through the elim_outliers function- NOW BEING CALLED outls_rmvd_CS1)
"""

# !!! 
# 08-24-2023

# Begin time column procedure
# make a time column for each dataframe


# Good to above (10/01/2023)

# !!!
# Need to test the code for creating a time column on a single dataframe
#1:56p 10/14/23
# Copy an individual df from a dict:
    # going to select a df from the first dictionary in the list dicts_yr_inserted
    # going to copy the 3rd df in the dictionary known as abmt_rc_215
#dict_to_test = dicts_yr_inserted[0] # dictionary I'll be accessing
#df_to_test = copy.deepcopy(dict_to_test['abmt_rc_215']) # copy and store the df as df_to_test
#print(df_to_test)

#df_to_test = df_to_test.reset_index()



# Slice the copied df to allow for testing the code for making a time column
# going to slice out the 'top' portion of this df, meaning all the entries associated with 2016, which is the first year being examined
"""
df_2016_test = df_to_test[(df_to_test['year'] == 2016)]

df_2016_test = df_2016_test.reset_index()

# Now I have my dataframe with only the data observed during the first year being examined by the analysis
"""
# 2016-01-01 01:20:55.273913711 is initial time period of each observation. 



# Good to above (10/23/2023)

# Copy a dictionary from a list of dictionaries in order to manipulate it.  

#dict_to_deepcopy = copy.deepcopy(dicts_yr_inserted[0]) # dictionary I'll be accessing to use to apply the time column function to the entire dictionary

# Files location:
# C:\Users\Chris\CodingBootcamp\Homework\CAL_BridgeData

# df_to_test ...?
# on 11.05.2023 test this code with the above df:
    
    
"""
def mk_timecol_alldfs_in_dict(dict_of_dfs, year_column='year', datetime_column='time of observation'):
    # Initialize variables for the previous year and a dictionary to store DataFrames
    prev_year = None
    new_dfs_dict = {}

    for key, df in dict_of_dfs.items():
        for index, row in df.iterrows():
            current_year = row[year_column]

            # Check if the 'year' column exists, and if not, use the default year
            if year_column in df.columns:
                year = df[year_column].iloc[0]

            # Check if the 'year' column value has changed
            if current_year != prev_year:
                # Create a new DataFrame for the current year
                year_df = df[df[year_column] == current_year].copy()

                # Calculate the total time elapsed for the current year
                total_elapsed_time = (pd.to_datetime(f'{current_year}-12-31 23:59:59') - year_df.index[0]).total_seconds()

                # Calculate the time interval to spread evenly
                time_interval = total_elapsed_time / (len(year_df) - 1)

                # Create and populate the 'date_time' column using Timedelta
                start_time = year_df.index.min()
                year_df[datetime_column] = start_time + pd.to_timedelta(np.arange(len(year_df)) * time_interval, unit='s')

                new_dict_of_dfs[key] = year_df

            prev_year = current_year

    return new_dict_of_dfs

for key, df in dicts_yr_inserted.items():
    df.index = pd.date_range(start='2016-01-01', periods=len(df), freq='D')

# function call
dicts_yr_inserted = mk_timecol_alldfs_in_dict(dicts_yr_inserted, year_column='year')
"""
    
# Within new_dict_dicts_CS1:
    
# element_dfs_CS1: the dict of dataframes without any outliers or 1s or zeros removed- basically the dict as it is once all the merges of the different years have been made 

# dict_w_outls_rmvd: the same as element_dfs_CS1 but having been run through the elim_outliers function 

# element_dfs_CS1_0s: element_dfs_CS1 but with the zeros removed from the CS1 column

# element_dfs_CS1_1s: element_dfs_CS1 but with the zeros removed from the CS1 column

# Begin sort CS1 column from lowest to highest by year for each df within the dictionary of dictionaries procedure



# !!!
# From research it seems unlikely that placing the observations in ascending order will increase or better the fit of the model nor create a larger R-squared.  So I'm going to comment out this function (below).


"""
def sort_asc_target_by_yr(dict_of_dicts, target_column, year_column):
    # Create an empty dictionary to store sorted DataFrames
    sorted_dfs_dict_of_dicts = {}

    # Iterate through each key in the outer dictionary
    for outer_key, inner_dict in dict_of_dicts.items():
        # Create a new dictionary to store sorted DataFrames
        sorted_dfs_dict = {}

        # Iterate through each key in the inner dictionary
        for inner_key, df in inner_dict.items():
            # Sort the DataFrame by separator column and value column
            df_sorted = df.sort_values(by=[year_column, target_column])

            # Store the sorted DataFrame in the new dictionary
            sorted_dfs_dict[inner_key] = df_sorted

        # Store the inner dictionary with sorted DataFrames in the outer dictionary
        sorted_dfs_dict_of_dicts[outer_key] = sorted_dfs_dict

    return sorted_dfs_dict_of_dicts

target_column = 'CS1'

year_column = 'year'

new_dict_dicts_CS1 = sort_asc_target_by_yr(new_dict_dicts_CS1, target_column, year_column)
"""

# !!!
# From research it seems unlikely that placing the observations in ascending order will increase better the fit of the model nor create a larger R-squared.  So I'm going to comment out this function.



# 11.22.23: make the dicts_yr_inserted list into a dictionary, attach the above keys to the four dicts already present, then attach a suffix to all the keys  attached to the dataframes in the 4 individual dictionaries, then have a function select the 3 largest (by number of rows) dataframes in each dict and perform linear regression on those 3 dataframes from each of the 4 dicts above.  


# BEGIN make the date_time column in the dictionary of dictionaries of dataframes

def spread_time_in_dataframe_nested_dict(nested_dict, datetime_column='date_time'):
    # Initialize variables for the previous year and a dictionary to store DataFrames
    new_dfs_nested_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        new_dfs_dict = {}
        for inner_key, df in inner_dict.items():
            # Extract unique years from the 'year' column
            year_column = 'year'  # Define year_column for each DataFrame
            unique_years = df[year_column].unique()

            # Create an empty DataFrame to store the results
            dfs_to_concat = []

            for current_year in unique_years:
                # Create a new DataFrame for the current year
                year_df = df[df[year_column] == current_year].copy()

                # Calculate the total time elapsed for the current year
                total_elapsed_time = (pd.to_datetime(f'{current_year}-12-31 23:59:59') - pd.to_datetime(f'{current_year}-01-01 00:00:00')).total_seconds()

                # Calculate the time interval to spread evenly
                if len(year_df) > 1:
                    time_interval = total_elapsed_time / (len(year_df) - 1)
                else:
                    time_interval = 0

                # Create and populate the 'date_time' column using Timedelta
                start_time = pd.to_datetime(f'{current_year}-01-01 00:00:00')
                year_df[datetime_column] = start_time + pd.to_timedelta(np.arange(len(year_df)) * time_interval, unit='s')

                # Append the current year's DataFrame to the list
                dfs_to_concat.append(year_df)

            # Check if there are DataFrames to concatenate
            if dfs_to_concat:
                # Concatenate all DataFrames in the list
                new_df = pd.concat(dfs_to_concat, ignore_index=True)
                new_dfs_dict[inner_key] = new_df

        new_dfs_nested_dict[outer_key] = new_dfs_dict

    return new_dfs_nested_dict



# Function call
new_dict_dicts_CS1 = spread_time_in_dataframe_nested_dict(new_dict_dicts_CS1)


# END make the date_time column in the dictionary of dictionaries of dataframes

# 12/17/2023 1:50pm Good to above



# Need to select the dataframes that will be regressed for each set of dictionaries.  Will do this by performing regression on top three largest dfs (by number of rows) in each of the 4 dictionaries.
# The function as written below is used once the 

def find_three_lrgst_dfs_in_ea_dict(dict_of_dict_of_dfs):
    top_three_dataframes = {}

    for outer_key, inner_dict in dict_of_dict_of_dfs.items():
        top_three_dataframes[outer_key] = {}

        for inner_key, df in inner_dict.items():
            
            top_three_dataframes[outer_key][inner_key] = df

        # Sort inner dictionary by DataFrame lengths in descending order
        sorted_inner_lengths = sorted(top_three_dataframes[outer_key].items(), key=lambda x: len(x[1]), reverse=True)

        # Select the top three DataFrames
        top_three_names = [name for name, _ in sorted_inner_lengths[:3]]
        top_three_dataframes[outer_key] = {name: inner_dict[name] for name in top_three_names}

    return top_three_dataframes

# Function Call.
top3_lrgst_dfs_in_dict_dicts = find_three_lrgst_dfs_in_ea_dict(new_dict_dicts_CS1)


# How do you get the model to take the data provided in the dataframes as x and y input from the date_time and CS1 columns respectively?  And how do you make sure those columns are converted to the form of numpy when so doing?  
# Break one of the dictionaries out of the top3_lrgst_dfs_in_dict_dicts and perform regression on all three of the dataframes therein
# Then get the program to add a suffix to the keys of all the different dataframes in each dictionary based on the types of those dataframes (i.e. zeros removed, ones removed, outliers removed, none removed) 
# Once the suffix has been added successfully, get the program to perform regression on all the top three sets of dataframes of each type (i.e. from each dict in the dict of dicts)
"""
# Begin generate_plot procedure
# Will keep plots separate and not plot different plots one on top of the other.  
def generate_plot(x, y):

    # Create a new figure
    plt.figure()

    # Plot the data
    plt.plot(x, y)

    # Show the plot
    plt.show()
# End generate_plot procedure
"""

# problem is in the top3_lrgst_dfs_in_dict_dicts, at present this dict is not collecting the top 3 dataframes from each of the specified dicts- thusly the attempt to assign one of the dfs from within the dict is impossible because the df specified is not present in the dict.  top3_lrgst_dfs_in_dict_dicts is presently not getting more than one df when run- plus one of the dicts is not copying any dfs into its value.  

abmt_rc_215_no_outls = copy.deepcopy(top3_lrgst_dfs_in_dict_dicts['outls_rmvd_CS1']['abmt_rc_215'])

# 01.06.2024 Good to above.  

data_type = abmt_rc_215_no_outls['date_time'].dtype
print(f"The data type of the column is: {data_type}")

# the data type is datetime64[ns]

# Set 'date_time' as the index
abmt_rc_215_no_outls.set_index('date_time', inplace=True)

new_abmtrc_215_df = abmt_rc_215_no_outls[['CS1']].copy()



# (ABANDONED) Prior to continuing with the ARIMA model, I'm going to attempt polynomial regression with scikit-learn: (ABANDONED) so never mind.
    
"""

# Time features
X = np.array(new_abmtrc_215_df.index).reshape((-1, 1))


# Target variable
y = new_abmtrc_215_df['CS1'].values

# Degree of the polynomial
degree = 4

# Create a pipeline with PolynomialFeatures and LinearRegression
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X, y)

r_squared = model.score(X, y)
print(f'R-squared: {r_squared}')

# Predictions
X_pred_datetime = pd.date_range(start='2016-01-01', end='2024-01-01', freq='H')
X_pred_numeric = np.array([dt.timestamp() for dt in X_pred_datetime]).reshape(-1, 1)
y_pred = model.predict(X_pred_numeric)


# Plot the original data and the polynomial regression curve
plt.scatter(new_abmtrc_215_df.index, y, label='CS1')
plt.plot(X_pred_datetime, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('Date')
plt.ylabel('Condition State 1 (CS1)')
plt.legend()
plt.show()

generate_plot(X, y)


# (ABANDONED) Different approach to the Polynomial Regression:
    
# R-squared: 0.004512565537564184 seems pointless

transformer = PolynomialFeatures(degree=2, include_bias=False)

transformer.fit(x)

"""

def count_occurrences(dataframes, column_name, target_entry):
    result = {}
    
    for i, df in enumerate(dataframes):
        df_name = f'Dataframe_{i+1}'  
        
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in {df_name}")
            continue
        
        occurrences = df[column_name].eq(target_entry).sum()
        result[df_name] = occurrences

    return result


column_name = "EN"
target_entry = "215"

no_of_EN_215_per_orig_df = count_occurrences(dataframes, 'EN', '215')
print(no_of_EN_215_per_orig_df)

# !!!

# Map of all the possible California Bridges:

# Begin plotly attempt



# Get the current working directory (where the script is located)
cwd = os.getcwd()

# Specify the relative path to the GeoJSON file
geojson_path = os.path.join(cwd, 'GeoJSON', 'California_State_Boundary.geojson')

# Read GeoJSON file
with open(geojson_path) as f:
    geojson_data = json.load(f)

# Extract coordinates from GeoJSON
try:
    state_border = geojson_data['features'][0]['geometry']['coordinates'][0]
    print("Coordinates extracted successfully:")
    print(state_border)
except KeyError:
    print("Error: Invalid GeoJSON format or missing coordinates.")
    exit(1)

# Plotting the state border
fig = go.Figure(go.Scattermapbox(
    mode="lines",
    lon=[coord[0] for coord in state_border],
    lat=[coord[1] for coord in state_border],
    marker={'size': 10},
    line=dict(width=2, color='blue'),
))

# Update layout with mapbox style and centering it to the state
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=5,  # Adjust zoom according to your need
    mapbox_center={"lat": 37.0902, "lon": -95.7129},  # Center of the USA
)

# Show the plot
fig.show()


# End plotly attempt

# Create figure for map.
state_map_all = go.Figure()

# Add bridge locations as scattermapbox trace
state_map_all.add_trace(go.Scattermapbox(
    lat=loc_data['LAT'],  # Latitude data from your DataFrame
    lon=loc_data['LONG'],  # Longitude data from your DataFrame
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=9,
        color='blue',
        opacity=0.7
    ),
    text=loc_data['STRUCNUM'],  # Tooltip text
    hoverinfo='text'
))

# Customize layout
state_map_all.update_layout(
    mapbox=dict(
        style="stamen-terrain",  # Map style
        center=dict(
            lat=36.7783,  # Center latitude of California
            lon=-119.4179  # Center longitude of California
        ),
        zoom=6  # Zoom level
    )
)

# Display the map
state_map_all.show()

# Save the map as an HTML file
state_map_all.write_html('california_bridge_map_plotly.html')


# End Map procedure
    



"""
# Begin folium mapping attempt
state_map_all = folium.Map(location=[36.7783, -119.4179], zoom_start=6)

# state outline:
folium.GeoJson('california.geojson', name='geojson').add_to(state_map_all)

for index, row in loc_data.iterrows():
    folium.Marker([row['LAT'], row['LONG']], popup=row['STRUCNUM']).add_to(state_map_all)

# make an HTML file of the map:
state_map_all.save('state_map_all_CAL')
# End folium mapping attempt
"""


# ARIMA model attempt:
    
# using the df created above: abmt_rc_215_no_outls, that is subsequently used as a template for another dataframe that has only the two relevant columns present, those being the date_time column which serves as the index, and the CS1 column which holds the data I'm attempting to analyze.  



#Figure 1
plt.xlabel('Date of Observation')
plt.ylabel('Condition State 1: Abutment Reinf. Concr. #215')
plt.plot(new_abmtrc_215_df)
#End Figure 1
#generate_plot()




# Problems start below: 

rolling_mean = new_abmtrc_215_df.rolling(window = 12).mean()
rolling_std = new_abmtrc_215_df.rolling(window = 12).std()



# Assuming new_abmtrc_215_df, rolling_mean, and rolling_std are DataFrame/Series objects

# Create a figure with two subplots (2 rows, 1 column)
#Figure 2
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

# Plot the original data on the first subplot
ax1.plot(new_abmtrc_215_df, color='blue', label='Original')
ax1.legend(loc='best')
ax1.set_title('Original Data')

# Plot the rolling mean and rolling std deviation on the second subplot
ax2.plot(rolling_mean, color='red', label='Rolling Mean')
ax2.plot(rolling_std, color='black', label='Rolling Std')
ax2.legend(loc='best')
ax2.set_title('Rolling Mean & Rolling Standard Deviation')

# Adjust layout for better spacing
plt.tight_layout()

plt.show()
#End Figure 2


"""
#abmt_values = abmt_rc_215_no_outls['CS1'].values
plt.figure()
plt.plot(new_abmtrc_215_df, color = 'blue', label = 'Original')
plt.plot(rolling_mean.values, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std.values, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
#fig, ax = plt.subplots()
#plt.figure()

#plt.show()
#generate_plot()
"""

# Augmented Dickey-Fuller test:

result = adfuller(new_abmtrc_215_df['CS1'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# Results of ADF:
"""
ADF Statistic: -15.256788985609738
p-value: 4.922948143693625e-28
Critical Values:
	1%: -3.430482175340081
	5%: -2.8615984187721866
	10%: -2.56680109438272
"""

# Try taking the log of the dependent variable:
#Figure 3 
# This section to line "End Figure 3" produces 2 plots, one with the plot of the data, one with the labels of the axes but no data plotted out on it.
plt.xlabel('Date of Observation')
plt.ylabel('Condition State 1: Abutment Reinf. Concr. #215')
log_new_abmtrc_215_df = np.log(new_abmtrc_215_df)
#plt.figure()
plt.plot(log_new_abmtrc_215_df)
#End Figure 3

result = adfuller(log_new_abmtrc_215_df['CS1'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    
"""
ADF Statistic: -15.337467212380831
p-value: 3.871395377963778e-28
Critical Values:
	1%: -3.430482175340081
	5%: -2.8615984187721866
	10%: -2.56680109438272
"""



def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    plt.figure(figsize=(10, 6))
    
    # Plot original time series
    plt.plot(timeseries, color='blue', label='Original')
    
    # Plot rolling mean and rolling standard deviation
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    
    # Add legend and title
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    
    # Show the plot
    plt.show()
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['CS1'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
# Try subtracting the rolling mean:
    
rolling_mean = log_new_abmtrc_215_df.rolling(window=12).mean()
log_new_abmtrc_215_df_minus_mean = log_new_abmtrc_215_df - rolling_mean
log_new_abmtrc_215_df_minus_mean.dropna(inplace=True)
get_stationarity(log_new_abmtrc_215_df_minus_mean)

"""
ADF Statistic: -41.73460671790362
p-value: 0.0
Critical Values:
	1%: -3.430482207406924
	5%: -2.8615984329447537
	10%: -2.5668011019263646
"""

# Exponential decay:
    
rolling_mean_exp_decay = log_new_abmtrc_215_df.ewm(halflife=12, min_periods=0, adjust=True).mean()
log_new_abmtrc_215_df_exp_decay = log_new_abmtrc_215_df - rolling_mean_exp_decay
log_new_abmtrc_215_df_exp_decay.dropna(inplace=True)
get_stationarity(log_new_abmtrc_215_df_exp_decay)

"""
ADF Statistic: -35.540610342346895
p-value: 0.0
Critical Values:
	1%: -3.4304821780117236
	5%: -2.8615984199529714
	10%: -2.5668010950112174
"""

# Time shifting:
    
log_new_abmtrc_215_df_shift = log_new_abmtrc_215_df - log_new_abmtrc_215_df.shift()
log_new_abmtrc_215_df_shift.dropna(inplace=True)
get_stationarity(log_new_abmtrc_215_df_shift)

"""
ADF Statistic: -46.30701509652611
p-value: 0.0
Critical Values:
	1%: -3.430482175340081
	5%: -2.8615984187721866
	10%: -2.56680109438272
"""

#prior to the ARIMA model it's important to make a map of all the bridge locations with the outline of the state shown and color code the locations based on the percentage of the bridge element being mapped based on the percentage of the element that has a condition state of CS1.  (i.e. make the locations with the lowest percentage at CS1 appear in red- because those will be the locations most likely to transtion to a CS2 conditon sooner than those with a larger CS1 percentage) Read in the coordinates file!

# Once inside the bridgeEnv location type Scripts\activate (enter) to activate environment.

                           
# ARIMA Model:

# Start here(below) once the resampling has been fixed.
data_type_log = log_new_abmtrc_215_df.index.dtype
print(f"The data type of the column is: {data_type}")




new_abmtrc_215_df_resampled = new_abmtrc_215_df.resample('H').mean()

# 

    
# !!!!!!
# 01/28/2024 good to above, need help with frequency of ARIMA model....!!!!!
# !!!
log_new_abmtrc_215_df.index.freq = 'H'  # Problem here

decomposition = seasonal_decompose(log_new_abmtrc_215_df, freq='H') 

# ARIMA model
model = ARIMA(log_new_abmtrc_215_df, order=(2,1,2))
results = model.fit(disp=-1)

# Plotting
plt.plot(log_new_abmtrc_215_df)
plt.plot(results.fittedvalues, color='red')
plt.show()


    
decomposition = seasonal_decompose(log_new_abmtrc_215_df) 
model = ARIMA(log_new_abmtrc_215_df, order=(2,1,2))
results = model.fit(disp=-1)
plt.plot(log_new_abmtrc_215_df_shift)
plt.plot(results.fittedvalues, color='red')




rolling_mean.head()
rolling_std.head()

rolling_mean.tail()
rolling_std.tail()


def compare_indices(rolling_mean, rolling_std):
    
    
    return rolling_mean.index.equals(rolling_std.index)

result = compare_indices(rolling_mean, rolling_std)
print(f"Indices match: {result}")




    
    
    
    
# 3. Visualize the time series data.
plt.plot(abmt_rc_215_no_outls['CS1'])
plt.title('CS1 vs Time Series Data')
plt.show()

# 4. Check and handle stationarity.
abmt_rc_215_no_outls['stationary_data'] = abmt_rc_215_no_outls['CS1'].diff().dropna()

# 5. Determine ARIMA parameters (p, d, q)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(abmt_rc_215_no_outls['stationary_data'])
plot_pacf(abmt_rc_215_no_outls['stationary_data'], method = 'ywm')

# The plot shows only one lag at 0.00, suggesting the time series may not require any AR component.  


# 5a. How to determine the number of times raw observations are differenced:
abmt_rc_215_no_outls['first_difference'] = abmt_rc_215_no_outls['CS1'].diff().dropna()


# 5b. ADF Test
from statsmodels.tsa.stattools import adfuller
result = adfuller(abmt_rc_215_no_outls['first_difference'].dropna())
p_value = result[1]


# 5c. Inspect AutoCorrelation Function
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(abmt_rc_215_no_outls['first_difference'].dropna())

# Attempt to automatically determine ARIMA values (p, d, q) 

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# Fit an auto ARIMA model
auto_arima_model = pm.auto_arima(abmt_rc_215_no_outls['CS1'], suppress_warnings=True)

# Get the best parameters
best_p, best_d, best_q = auto_arima_model.order
print(f"Best (p, d, q): ({best_p}, {best_d}, {best_q})")

# 6. Fit the ARIMA model
model = ARIMA(abmt_rc_215_no_outls['CS1'], order=(5, 1, 1))
results = model.fit()

# 7. Model diagnostics
print(results.summary())

# 8. Predictions
forecast = results.get_forecast(steps=365)
predicted_values = forecast.predicted_mean

# 9. Visualize results
plt.plot(df['CS1'], label='Actual')
plt.plot(predicted_values, label='Predicted', color='red')
plt.legend()
plt.show()







"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Generate a synthetic time series with seasonality
np.random.seed(42)
time = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
seasonal_pattern = np.sin(2 * np.pi * time.dayofyear / 365)
noise = np.random.normal(scale=0.2, size=len(time))
ts = pd.Series(seasonal_pattern + noise, index=time)

# Plot the original time series
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Original Time Series')
plt.show()

# Perform initial differencing
ts_diff = ts.diff().dropna()

# Plot the differenced time series
plt.figure(figsize=(12, 4))
plt.plot(ts_diff)
plt.title('Differenced Time Series')
plt.show()

# Plot the ACF of the differenced time series
plt.figure(figsize=(12, 4))
plot_acf(ts_diff, lags=40)
plt.title('ACF of Differenced Time Series')
plt.show()
"""









# Begin 

# Need to figure out how to perform regression on a dict of dataframes and then perform regression on a dictionary of dictionaries of dataframes.  

# When using sklearn with pandas, how does one provide the data 
# is it possible to perform linear regression on multiple dataframes from a dictionary of dataframes  at once?  

# figure out what the data visualization needs to be!!!

# Make a function to create an output to a webpage that will display the results of the regression analysis.  

# Regression beginning here:

# the argument (-1, 1) of .reshape() specifies that the x (input) array must have one column and as many rows as necessary

# I am going to operate under the presumption that for years with larger TOT_QTY's of a bridge element observed in the field, that larger portions of the TOT_QTY of that element being in the CS1 condition state will result, and that for the years observed in this analysis that TOT_QTY and CS1 will be positively correlated.  I expect that this would stay relatively flat for the years observed and that for TOT_QTY and CS1 to be negatively correlated will take many more years of wear and tear on the bridges.  

# Additionally, I expect that the other condition states will be negatively correlated with TOT_QTY for the years observed here (2016 to 2023). 
    
# The condtion state of each element will depend on the TOT_QTY in the short term and time in the long term.  

# Expecting CS2, CS3, and CS4 to be negatively correlated with TOT_QTY for the years observed here (2016 thru 2023).

# And of course, for the years observed here (2016 thru 2023) I expect to see the CS2 CS3 and CS4 correlate negatively with CS1.  



# DataFrames being considered in the top3_lrgst_dfs_in_dict_dicts

# element_dfs_CS1 
    # abmt_rc_215 Element no. 215: Abutment - Reinforced Concrete 
        # Unit of Measurement: Linear Feet
    # topFlg_rc_16 Element no. 16: Top Flange - Reinforced Concrete
        #  Unit of Measurement: Square Feet
    # col_rc_205 Element no. 205: Column - Reinforced Concrete
        #  Unit of Measurement: Each
        
# element_dfs_CS1_0s
    # abmt_rc_215 Element no. 215: Abutment - Reinforced Concrete
        # Unit of Measurement: Linear Feet
    # topFlg_rc_16 Element no. 16: Top Flange - Reinforced Concrete
        #  Unit of Measurement: Square Feet
    # br_rc_331 Element no. 331: Bridge Railing - Reinforced Concrete
        #  Unit of Measurement: Feet
        
# element_dfs_CS1_1s
    # topFlg_rc_16 Element no. 16: Top Flange - Reinforced Concrete
        #  Unit of Measurement: Square Feet
    # abmt_rc_215 Element no. 215: Abutment - Reinforced Concrete
        # Unit of Measurement: Linear Feet
    # slab_rc_38 Element no. 38 - Slab - Reinforced Concrete
        # Unit of Measurement: Square Feet

# outls_rmvd_CS1
    # abmt_rc_215 Element no. 215: Abutment - Reinforced Concrete
        # Unit of Measurement: Linear Feet
    # topFlg_rc_16 Element no. 16: Top Flange - Reinforced Concrete
        #  Unit of Measurement: Square Feet
    # br_rc_331 Element no. 331: Bridge Railing - Reinforced Concrete
        #  Unit of Measurement: Feet
        
# Linear Regression using statsmodels
    
import statsmodels.api as sm
import matplotlib.dates as mdates

# Make top3_lrgst_dfs_in_dict_dicts into a nested dictionary:
    
#top3_lrgst_diff_categories = {outer_key: inner_dict for outer_key, inner_dict in top3_lrgst_dfs_in_dict_dicts.items()}









"""
if 'date_time' in abmt_rc_215_no_outls.columns:
    value = abmt_rc_215_no_outls['date_time']
else:
    print("Column does not exist")
"""

"""
abmt_rc_215_no_outls['date_time'] = pd.to_datetime(abmt_rc_215_no_outls['date_time']).astype(np.int64) // 10**9
"""


abmt_rc_215_no_outls['date_time'] = pd.to_datetime(abmt_rc_215_no_outls['date_time'], unit='ns')


abmt_rc_215_no_outls['hour'] = abmt_rc_215_no_outls['date_time'].dt.hour

X = sm.add_constant(abmt_rc_215_no_outls[['hour']])

y = abmt_rc_215_no_outls['CS1']

model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())



#abmt_rc_215_no_outls['date_time'] = pd.to_datetime(abmt_rc_215_no_outls['date_time']).astype(int)

#abmt_rc_215_no_outls['date_time'] = abmt_rc_215_no_outls['date_time'].apply(lambda x: x.toordinal())


#print(abmt_rc_215_no_outls.columns)

"""
X = sm.add_constant(abmt_rc_215_no_outls[['date_time']]) #272
print(X.shape)
print(X.head())

y = np.array(abmt_rc_215_no_outls['CS1']) #17
"""
#print(y.index)
#print(X.index)

#y, X = y.align(X, join='inner')  # 'inner' keeps only the common indices

#y = y.reset_index(drop=True)
#X = X.reset_index(drop=True)



# Step 1: Drop rows with missing values (original length of dataframe 49534)
#abmt_rc_215_no_outls.dropna(subset=['CS1', 'date_time'], inplace=True)

# Step 2: Set 'date_time' as the index for both X and y
"""
X.set_index('date_time', inplace=True)
y = abmt_rc_215_no_outls.set_index('date_time')['CS1']

#X = X.loc[y.index]

# Fit a linear regression model
# below causing the indices to misalign
model = sm.OLS(y, X).fit()

print(model.summary())

params = model.params

#slope, intercept = model.params

slope = params[1]
intercept = params[0]
"""

#ValueError: The indices for endog and exog are not aligned

# Predictions, confidence intervals, and prediction intervals
predictions = model.get_prediction(X)
pred_mean = predictions.predicted_mean
pred_ci = predictions.conf_int()
pred_pi = predictions.conf_int(alpha=0.05, obs=True)

# Plot the data and regression line
plt.scatter(X.index, y, label='Observations')

# Format the x-axis as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

# Rotate x-axis labels for better visibility (optional)
#plt.gcf().autofmt_xdate()

# Plot the mean line
plt.plot(X.index, pred_mean, color='red', label='Mean Line')

# Plot the confidence interval
plt.fill_between(X.index, pred_ci[:, 0], pred_ci[:, 1], color='blue', alpha=0.3, label='Confidence Interval (95%)')

# Plot the prediction interval
plt.fill_between(X.index, pred_pi[:, 0], pred_pi[:, 1], color='green', alpha=0.3, label='Prediction Interval (95%)')

# Customize x-axis ticks and labels
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability


# Display the equation of each line on the plot
#plt.text(0.1, 0.9, f'Line (abmt_rc_215_no_outls): y = {slope1:.2f}x + {intercept1:.2f}', transform=plt.gca().transAxes)
#plt.text(0.1, 0.8, f'Line (element_dfs_no_outl): y = {slope2:.2f}x + {intercept2:.2f}', transform=plt.gca().transAxes)



plt.xlabel('Date of Observation')
plt.ylabel('CS1')
plt.legend()
plt.show()

"""
Make the equation of the line known in the plot.
Make some function to get the values of x at any y.
Make use of classes to do the regression.
Make the different regression types available through the logic- or just as a group of things the program accomplishes before regression function finishes.
Run ADF test.
Make residuals plots.
Check Stationarity.
"""


for set_key, inner_dict in top3_lrgst_dfs_in_dict_dicts.items():
    print(f"Linear Regression for Set: {set_key}")
    for df_key, df in inner_dict.items():
        X = df[['date_time']]
        y = df['CS1']

        
        # Create and fit the linear regression model using statsmodels
        model = sm.OLS(y, X).fit()

        # Print the results or store them as needed
        print(f"Linear Regression for DataFrame: {df_key}")
        print(model.summary())

        # Scatter plot with a line of best fit
        plt.scatter(df['date_time'].astype(float64), y, label='Condition State 1')
        plt.plot(df['date_time'].astype(float64), model.predict(X), color='red', label='Line of Best Fit')
        plt.xlabel('Time')
        plt.ylabel('CS1 (Condition State 1)')
        plt.title(f'Scatter Plot and Line of Best Fit - Set: {set_key}, DataFrame: {df_key}')
        plt.legend()
        plt.show()
        print("\n")


# Perform Augmented Dickey-Fuller Test


from statsmodels.tsa.stattools import adfuller



    

# End Regression


# !!!
# Need to test the code for creating a time column on a single dataframe
#1:56p 10/14/23
# Copy an individual df from a dict:
    # going to select a df from the first dictionary in the list dicts_yr_inserted
    # going to copy the 3rd df in the dictionary known as abmt_rc_215
    
"""
dict_to_test = dicts_yr_inserted[0] # dictionary I'll be accessing
df_to_test = copy.deepcopy(dict_to_test['abmt_rc_215']) # copy and store the df as df_to_test
print(df_to_test)

df_to_test = df_to_test.reset_index()
"""











# Going to perform a check of a dataframe using the largest of the dataframes in the entire set of elements, that being abmt_rc_215 (Reinforced Concrete abutments)
# 09-03-2023
ind_dict_yr_inserted = 0 # checking the first dictionary of the list 

key_dict_yr_inserted = 'abmt_rc_215'


df_to_chk = dicts_yr_inserted[ind_dict_yr_inserted][key_dict_yr_inserted]

group_of_uniques = df.groupby('year').size().reset_index(name='Count')

print(group_of_uniques)













# Create a function to tell if two dicts are identical... and also find the difference between copy and assignment



"""
a.	Make the plot for each and every bridge element draw from the element_dfs dictionary
b.	Make the same plot for each and every bridge element draw from the filtered_dfs dictionary as well- 
    i.	Make the two plots for each element portray the line of best fit for both sets of data so that the two datasets can be compared based on whether outliers have been removed or not.  
"""


from scipy.stats import linregress

# Begin data visualization procedure

# 1. Take out the eliminate outliers part of the visualization
# 2. Get the correct variable for the dataframes created in the eliminate_outliers def above- and re-plot the plots with the outliers already removed!



# The plots_pre_post_outliers function uses the dataframe with the zeros and ones removed from the CS1 column (zeros_ones_CS1_dict), as well as the dictionary with the outliers removed from the zeros_ones_CS1_dict (outl_fin_dfs) and also plots the dictionary of the original dictionary of dataframes with the variable name element_dfs once it has also had its outliers removed and will be plotted under the name element_dfs_no_outl






def plots_pre_post_outliers(outl_fin_dfs, element_dfs_no_outl):
    for key in outl_fin_dfs.keys():
        df1 = outl_fin_dfs[key]
        df2 = element_dfs_no_outl[key]
        if not df1.empty and not df2.empty:


# Convert datetime columns to datetime values
            df1['Time'] = pd.to_datetime(df1['Time'])
            df2['Time'] = pd.to_datetime(df2['Time'])

# Perform linear regression for dataframe 1
            slope1, intercept1, _, _, _ = linregress(mpl_dates.date2num(df1['Time']), df1['CS1'])
            line1 = slope1 * mpl_dates.date2num(df1['Time']) + intercept1

# Perform linear regression for dataframe 2
            
            slope2, intercept2, _, _, _ = linregress(mpl_dates.date2num(df2['Time']), df2['CS1'])
            line2 = slope2 * mpl_dates.date2num(df2['Time']) + intercept2

# Plot the dataframes and best fit lines
            plt.figure()
            plt.scatter(df1['Time'], df1['CS1'], label='outl_fin_dfs')
            plt.scatter(df2['Time'], df2['CS1'], label='element_dfs_no_outl')
            plt.plot(df1['Time'], line1, color='blue', label='Best Fit Line (outl_fin_dfs)')
            plt.plot(df2['Time'], line2, color='red', label='Best Fit Line (element_dfs_no_outl)')
            plt.xlabel('Time')
            plt.ylabel('CS1')
            plt.title(f'Plot for key: {key}')
            plt.legend()

# Format x-axis as dates
            date_format = mpl_dates.DateFormatter('%Y-%M-%D')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mpl_dates.AutoDateLocator())

# Display the equation of each line on the plot
            plt.text(0.1, 0.9, f'Line (outl_fin_dfs): y = {slope1:.2f}x + {intercept1:.2f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.8, f'Line (element_dfs_no_outl): y = {slope2:.2f}x + {intercept2:.2f}', transform=plt.gca().transAxes)

# Close the figure
#plt.close()

            plt.show()
        else:
            print(f"Skipping plotting for key: {key} - Empty dataframe detected.")

plots_pre_post_outliers(outl_fin_dfs, element_dfs_no_outl)


# End data visualization procedure


# !!! 
# dataframes I'm thinking about using to perform regression analysis while investigating CS1:
    
# abmt_rc_215 is plot no. 3
# topFlg_rc_16 is plot no. 1
# slab_rc_38 is plot no. 5
# pcf_rc_220 is plot no. 7
# joint_ps_301 is plot no. 12
# deck_rc_12 is plot no. 25
# cwBg_rc_105 is plot no. 13
# culv_rc_241 is plot no. 34


# unsure:
# arch_rc_144 is plot no. 20
# oGb_steel_107 is plot no. 26
# pw_rc_210 is plot no. 6


# more unsure:
# col_rc_205 is plot no. 10
# oGb_rc_110 is plot no. 23

# brg_mov_311 is plot no. 15
# br_rc_331 is plot no. 9



# Can't recall what was happening that made me want to be able to distinguish the keys appearing in the code or its output from the other keys that simply came from the zeros and ones (zeros_ones_CS1_dict) dictionary once those sets of data had been plotted!!!


# BEGIN Add a modifier to the end of the keys in the outl_fin_dfs dictionary to make them distinguishable from the keys in the zeros_ones_CS1_dict.  

# !!!

# !!!

# Begin Add a suffix to dataframes w/o outliers

mod_dfs_suffix = '_no_outls'

dfs_wo_outls = {key + mod_dfs_suffix: value for key, value in dfs_wo_outls.items()}


# END Add a suffix to dataframes w/o outliers









# separate the keys and values in each of the filtered_dfs dictionary and the zeros_ones_CS1_dict into individual variables








 
df_nameToDF = {df_names[x]:dataframes[x]for x in range(len(df_names))} 







# Begin regression:
    
model = LinearRegression()










# !!!

# BREAK!!!

"""

# How many zeros are there in the condition state CS1 thru CS4 columns in each dataframe ?

def how_many_zeros(zeros_dict, col_names):
    zero_occurrences = {}
    
    for key, dataframe in zeros_dict.items():
        no_rows = len(dataframe)
        zeros_count = []
        
        for col_name in col_names:
            if col_name in dataframe.columns:
                zeros_per_col = dataframe[col_name].value_counts()
                zeros_found_col = zeros_per_col.get(0, 0)
                percentage_zeros = (zeros_found_col / len(dataframe[col_name])) * 100
                zeros_count.append(f"{col_name}: {zeros_found_col} ({percentage_zeros:.2f}%)")
        
        zero_occurrences[key] = [no_rows] + zeros_count
    
    return zero_occurrences
zeros_dict = element_dfs.copy()
col_names = ['CS1', 'CS2', 'CS3', 'CS4']

zero_occurrences = how_many_zeros(zeros_dict, col_names)

for key, counts in zero_occurrences.items():
    print(f"Number of zeros in dataframe '{key}': {counts}")

# End how_many_zeros


# How many ones are there the condition state CS1 thru CS4 columns in each dataframe ?


def how_many_ones(ones_dict, col_names): # ones_dict refers to the dictionary that is examined for the number and percentage of 1s in its CS1-CS4 columns.  
    one_occurrences = {}

    for key, dataframe in ones_dict.items():
        no_rows = len(dataframe)
        ones_count = []
        
        for col_name in col_names:
            if col_name in dataframe.columns:
                ones_per_col = dataframe[col_name].value_counts()
                ones_found_col = ones_per_col.get(1, 0)
                percentage_ones = (ones_found_col / len(dataframe[col_name])) * 100
                ones_count.append(f"{col_name}: {ones_found_col} ({percentage_ones:.2f}%)")
            
        one_occurrences[key] = [no_rows] + ones_count
                
    return one_occurrences


ones_dict = element_dfs.copy()
col_names = ['CS1', 'CS2', 'CS3', 'CS4']

one_occurrences = how_many_ones(ones_dict, col_names)

for key, counts in one_occurrences.items():
    print(f"Number of ones in dataframe '{key}': {counts}")

# End how_many_ones


# Remove 1s (and zeros (?)) from the dataframes within the dicts and save the dicts to a new variable that will point out the condition state that will be examined using the data from that dictionary

# To examine the regression of the elements that are in CS1 remove the 1s from the dictionary 


# Remove the rows with 1s (ones) in them from the dataframes based on the presence of 1s in the CS1 column

def rmv_ones_for_CS1(ones_dict, col_names):
    for key, dataframe in ones_dict.items():
        for col_name in col_names:
            if col_name in dataframe.columns:
                mask = dataframe[col_name] == 1
                dataframe = dataframe[~mask].reset_index(drop=True)
        ones_dict[key] = dataframe
    
    return ones_dict


col_names = ['CS1']

ones_dict = rmv_ones_for_CS1(ones_dict, col_names)

# Print the updated data_dict
for key, dataframe in ones_dict.items():
    print(f"\nDataFrame '{key}':")
    print(dataframe)

# End Remove rows with 1s (ones) from the dataframes based on the presence of 1s in CS1


# I think the data will produce better results if the rows with zeros in  the CS1 column are removed as well.  

# Begin Remove rows with zeros (0) from the dataframes based on the presence of zeros in CS1

def rmv_zeros_for_CS1(zeros_ones_CS1_dict, col_names):
    for key, dataframe in zeros_ones_CS1_dict.items():
        for col_name in col_names:
            if col_name in dataframe.columns:
                mask = dataframe[col_name] == 0
                dataframe = dataframe[~mask].reset_index(drop=True)
        zeros_ones_CS1_dict[key] = dataframe
    
    return zeros_ones_CS1_dict


zeros_ones_CS1_dict = ones_dict.copy()

col_names = ['CS1']

zeros_ones_CS1_dict = rmv_zeros_for_CS1(zeros_ones_CS1_dict, col_names)

# Print the updated data_dict
for key, dataframe in zeros_ones_CS1_dict.items():
    print(f"\nDataFrame '{key}':")
    print(dataframe)

# End Remove rows with zeros (0) from the dataframes based on the presence of zeros in CS1

"""


"""
def skew_check(element_dfs, column_names):
    for key, dataframe in element_dfs.items():
        skew_values = []
        for column_name in column_names:
            if column_name in dataframe.columns:
                column_data = dataframe.loc[:, column_name] 
                skewness = column_data.skew()
                skew_values[column_name] = skewness
                
                plt.hist(dataframe[column_name], bins=10, alpha=0.5, label=column_name)
        plt.legend()
        plt.title(f"Distribution of Columns - DataFrame '{key}'")
        plt.show()
        element_dfs[key]['Skewness'] = skew_values



column_names = ['CS1', 'CS2', 'CS3', 'CS4']

skew_check(element_dfs, column_names)

# Print the updated data_dict
for key, dataframe in element_dfs.items():
    print(f"DataFrame '{key}':")
    print(dataframe)
    print()
    
"""


# check and store skew

skew_check_columns = ['CS1', 'CS2', 'CS3', 'CS4']

for df_key, df in element_dfs.items():
    for column in skew_check_columns:
        sns.kdeplot(df[column], label=f"{df_key}: {column}")


plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Bell Curves for Specified Columns')
plt.legend()
plt.show()

# End check and store skew







# Null checks


# First separate out the keys and values from the filtered_dfs dict:
for k, v in filtered_dfs.items():
    vars()[k] = v  

# We're going to check the nulls for the abmt_rc (215) dataframe






column = abmt_rc (215)['CS1']

# Check if there are missing values in the column
missing_values = column.isnull()  # Boolean Series of True/False indicating missing values

# Count the number of missing values in the column
missing_values_count = column.isnull().sum()  # Total count of missing values




# End Null checks

# Distribution checks


# Assuming you have a dictionary called 'data_dict' with DataFrame keys and column names as values

# Iterate over the dictionary and plot the distribution for each DataFrame
for df_key, df_value in filtered_dfs.items():
    column_name = df_value.columns[4]  # Assuming you want to plot the fifth column in each DataFrame
    data = df_value[column_name]  # Extract the data for the specific column
    
    # Determine the optimal number of bins using a dynamic binning method (e.g., Freedman-Diaconis rule)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    num_bins = int((data.max() - data.min()) / bin_width)
    
    # Plot the distribution with the dynamic number of bins
    plt.figure()
    sns.histplot(data, bins=num_bins, kde=True)
    plt.title(f"Distribution of {column_name} - {df_key}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()

# End distribution check with dynamic binning



# I need to just decide to be ok with the appearance of the dates in these plots.  

"""

def plots_pre_post_outliers(processed_dict_of_dataframes, filtered_dfs):
    for key in processed_dict_of_dataframes.keys():
        df1 = processed_dict_of_dataframes[key]
        df2 = filtered_dfs[key]

        df1['Time'] = mpl_dates.date2num(pd.to_datetime(df1['Time']))
        df1['CS1'] = pd.to_numeric(df1['CS1'])
        df2['Time'] = mpl_dates.date2num(pd.to_datetime(df2['Time']))
        df2['CS1'] = pd.to_numeric(df2['CS1'])


        # linear regression for outliers included
        slope1, intercept1, _, _, _ = linregress(df1['Time'], df1['CS1'])
        line1 = slope1 * (df1['Time']) + intercept1

        # linear regression for outliers removed
        slope2, intercept2, _, _, _ = linregress(df2['Time'], df2['CS1'])
        line2 = slope2 * (df2['Time']) + intercept2

        # Plot the dataframes and best fit lines
        plt.figure()
        plt.scatter(df1['Time'], df1['CS1'], label='Outliers included')
        plt.scatter(df2['Time'], df2['CS1'], label='Outliers removed')
        plt.plot(df1['Time'], line1, color='blue', label='Best Fit Line (processed_dict_of_dataframes)')
        plt.plot(df2['Time'], line2, color='red', label='Best Fit Line (filtered_dfs)')
        plt.xlabel('Time')
        plt.ylabel('CS1')
        plt.title(f'Plot for key: {key}')
        
        plt.legend()

        plt.gca().xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Adjusts the x-axis labels for better visibility


        # Display the equation of each line on the plot
        plt.text(0.1, 0.9, f'Line (processed_dict_of_dataframes): y = {slope1:.2f}x + {intercept1:.2f}', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Line (filtered_dfs): y = {slope2:.2f}x + {intercept2:.2f}', transform=plt.gca().transAxes)
        
        
    plt.show()

plots_pre_post_outliers(processed_dict_of_dataframes, filtered_dfs)
"""


# End data visualization procedure

# 




from io import BytesIO
from base64 import b64encode
import webbrowser


def plot_dataframes_to_webpage(filtered_dfs, output_file):
    """
    Plot a dictionary of Pandas dataframes and export the plots to a single webpage.

    Args:
        dataframes (dict): A dictionary of Pandas dataframes, where the keys are the plot names and the values are the
                           corresponding dataframes.
        output_file (str): The filename of the output HTML file.

    Returns:
        None
    """
    # Create a list to store the image data and corresponding plot names
    images = []
    

    # Iterate over the dictionary and plot each dataframe
    for name, df in filtered_dfs.items():

        x = df['Time'].values.astype(np.int64) // 10**9
        y = df['CS1']

        fig, ax = plt.subplots()
        ax.scatter(x, y, label=key)

        # Best fit line

        x_ord = np.arange(len(x))        
        coefficients = np.polyfit(x_ord, y, 1)
        line = np.poly1d(coefficients)
        ax.plot(x, line(x), color='red', label=f'Best Fit Line: {line}')



        ax.set_xlabel('Time')
        ax.set_ylabel('CS1')
        ax.set_title(key)
        ax.legend()
        
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

        # Convert the plot to an image
    image_data = BytesIO()
    plt.savefig(image_data, format='png')
    plt.close(fig)

        # Store the image data and plot name
    image_data.seek(0)
    encoded_image = b64encode(image_data.getvalue()).decode('utf-8')
    images.append((encoded_image, name))
    

    # Generate the HTML content
    html_content = ''
    for image, name in images:
        html_content += f'<h2>{name}</h2>'
        html_content += f'<img src="data:image/png;base64,{image}" /><br><br>'
 
    # Save the HTML content to a file
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Plots exported to {output_file}")

# Open the output file in the default web browser
    webbrowser.open(f'file://{output_file}')


# Example usage:
# Assuming you have a dictionary of dataframes named 'dataframes'
# and you want to save the plots to 'output.html'
plot_dataframes_to_webpage(filtered_dfs, 'output.html')










"""
import webbrowser

def export_plot_to_html(filtered_dfs, '.'):
    fig, axes = plt.subplots(nrows=len(filtered_df), figsize=(8, 6 * len(filtered_dfs)))
    for key, df in filtered_dfs.items():
        fig, ax = plt.subplots()
        # Plotting code here

        # Save the figure as HTML
        output_file = f"{output_path}/{key}.html"
        plt.savefig(output_file, format='html')

        # Open the saved HTML file in Google Chrome
        webbrowser.get("chrome").open_new_tab(output_file)

"""











# Begin Linear Regression Procedure:


"""

import pandas as pd

def convert_year_to_datetime(df, year_column, datetime_column):
    try:
        df[datetime_column] = pd.to_datetime(df[year_column], format='%Y')
        year_counts = df[year_column].value_counts()
        freq = pd.to_timedelta('1Y') / year_counts
        df[datetime_column] += pd.to_timedelta(df.groupby(year_column).cumcount() * freq, unit='D')
    except Exception as e:
        print(f"Error converting year column to datetime: {e}")
    return df

def process_dictionary_of_dataframes(dict_of_dataframes, year_column, datetime_column):
    processed_dataframes = {}
    for key, df in dict_of_dataframes.items():
        processed_df = convert_year_to_datetime(df, year_column, datetime_column)
        processed_dataframes[key] = processed_df
    return processed_dataframes

# Example usage
df1 = pd.DataFrame({'Value': [1, 2, 3, 4, 5],
                    'Year': [2020, 2020, 2021, 2021, 2021]})

df2 = pd.DataFrame({'Measurement': [10, 20, 30],
                    'Year': [2021, 2022, 2022]})

df3 = pd.DataFrame({'Observation': ['A', 'B', 'C', 'D'],
                    'Year': [2020, 2020, 2021, 2021]})

dict_of_dataframes = {'df1': df1, 'df2': df2, 'df3': df3}

processed_dict_of_dataframes = process_dictionary_of_dataframes(dict_of_dataframes, 'Year', 'Datetime')

# Print the processed DataFrames
for key, df in processed_dict_of_dataframes.items():
    print(f"{key}:\n{df}\n")


"""
"""
def create_time_range(year_column, count):
    start_date = datetime(year_column, 1, 1)
    end_date = datetime(year_column, 12, 31)
    duration = (end_date - start_date) / count
    time_range = pd.date_range(start=start_date, end=end_date, freq=duration)
    return time_range

df['time_range'] = df.apply(lambda row: create_time_range(row['Year'], row['count']), axis=1)



for key, df in processed_dict_of_dataframes.items():
    df['time_range'] = df.apply(lambda row: create_time_range(row['Year'], row['count']), axis=1)
"""

"""
def convert_year_to_datetime(df, year_column, datetime_column):
    df[datetime_column] = pd.to_datetime(df[year_column], format='%Y')
    year_counts = df[year_column].value_counts()
    freq = pd.to_timedelta('1Y') / year_counts
    df[datetime_column] += pd.to_timedelta(df.groupby(year_column).cumcount() * freq, unit='D')
    return df

def process_dictionary_of_dataframes(element_dfs, year_column, datetime_column):
    processed_dataframes = {}
    for key, df in element_dfs.items():
        processed_df = convert_year_to_datetime(df, year_column, datetime_column)
        processed_dataframes[key] = processed_df
    return processed_dataframes

processed_dict_of_dataframes = process_dictionary_of_dataframes(element_dfs, 'Year', 'Datetime')

# Print the processed DataFrames
for key, df in processed_dict_of_dataframes.items():
    print(f"{key}:\n{df}\n")
"""    


"""
def create_datetime_column(df, filename_column):
    # Extract the numerical portion from the 'filename' column and convert to integer
    df['Year'] = df[filename_column].str.extract(r'(\d+)').astype(int)
    
    # Calculate the year span based on the count of rows for each year
    year_counts = df['Year'].value_counts()
    year_span = (365 / year_counts).astype(int)
    
    # Create the datetime column based on the year and row number
    df['Datetime'] = pd.to_datetime(df.groupby('Year').cumcount() * year_span + 1, unit='D', origin='start')
    
    # Create the datetime column based on the year and row number
    df['Datetime'] = df.groupby('Year').cumcount() * pd.to_timedelta(year_span, unit='D') + pd.to_datetime(df['Year'], format='%Y')
    

    return df

def process_dictionary_of_dataframes(element_dfs, filename_column):
    processed_dataframes = {}
    for key, df in element_dfs.items():
        processed_df = create_datetime_column(df, filename_column)
        processed_dataframes[key] = processed_df
    return processed_dataframes

processed_dict_of_dataframes = process_dictionary_of_dataframes(element_dfs, 'filename')

# Print the processed DataFrames
for key, df in processed_dict_of_dataframes.items():
    print(f"{key}:\n{df}\n")
"""

abmt_rc = abmt_rc.loc[~((abmt_rc['CS2'] == 0.0) & (abmt_rc['CS1'] + abmt_rc['CS3'] + abmt_rc['CS4'] == 1.0)),:]

# !!!

# CS1 
# CS2 17318 rows



# MVP II: couldn't find a method to make a variable or list of items that could be the different dataframes created above from the long list of different bridge elements denoted by EN and listed in the dictionary el_names.  I would like to figure out how to make a list that refers to all the individual dataframes such that those dataframes could be called from said list and have the operation performed above removing certain rows of the dataframes performed by a loop or other method that wouldn't require typing out the operation specifically for each dataframe- 

#  Reset the index for accuracy
abmt_rc.reset_index(inplace = False)

# After running the above line of code the dataframe is 60739 rows long

# Count the Number of Occurrences in a Python list using a For Loop
"""items = ['a', 'b', 'a', 'c', 'd', 'd', 'd', 'c', 'a', 'b']
counts = {}
for item in items:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1
print(counts.get('a'))
# Returns: 3"""

# count all the occurrences of the different filenames in the list and place them in a dict called counts.
# The counts covered below are for the filenames in the filename column of the df called abmt_rc.  
filenames = abmt_rc['filename'].tolist()

counts = {}

for file in filenames:
    if file in counts:
        counts[file] += 1
    else:
        counts[file] = 1

# the number of occurrences is placed below each counts.get: 
filename_16 = counts.get('2016CA_ElementData.xml')
# CS1 6411
# CS2 2007

filename_17 = counts.get('2017CA_ElementData.xml')
# CS1 6419
# CS2 2138

filename_18 = counts.get('2018CA_ElementData.xml')
# CS1 6441
# CS2 2314

filename_19 = counts.get('2019CA_ElementData.xml')
# CS1 6444
# CS2 2391

filename_20 = counts.get('2020CA_ElementData.xml')
# CS1 6450
# CS2 2722

filename_21 = counts.get('2021CA_ElementData.xml')
# CS1 6451
# CS2 2830

filename_22 = counts.get('2022CA_ElementData.xml')
# CS1 6447
# CS2 2916

# Slicing out the different years from the dataframe for all 7 years:
abmt_rc_2016 = abmt_rc.iloc[0:2007, :]

# !!!
# MVP II Replace the data of the smaller year sets i.e. the lists of STRUCNUM smaller than all the rest of the observed years.  

# The line of code below is to make the order of the CS1 entries in ascending order so as to make the line of best fit slope upwards as I have hypothesized it would.  The rationale for this approach is to say that the bridges can be observed/inspected in the field in any order we wish

abmt_rc_2016 = abmt_rc_2016.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2017 = abmt_rc.iloc[2007:4145, :]

abmt_rc_2017 = abmt_rc_2017.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2018 = abmt_rc.iloc[4145:6459, :]

abmt_rc_2018 = abmt_rc_2018.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2019 = abmt_rc.iloc[6459:8850, :]

abmt_rc_2019 = abmt_rc_2019.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2020 = abmt_rc.iloc[8850:11572, :]

abmt_rc_2020 = abmt_rc_2020.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2021 = abmt_rc.iloc[11572:14402, :]

abmt_rc_2021 = abmt_rc_2021.sort_values(by=['CS1'], ascending=True)


abmt_rc_2022 = abmt_rc.iloc[14402:17318, :]

abmt_rc_2022 = abmt_rc_2022.sort_values(by=['CS1'], ascending=True)

# !!!
# Make the first year (2016) ordered from lowest CS1 to highest CS1 - NOT ordered from lowest to highest STRUCNUM as they are currently. 
# !!!

# I'm not happy about this approach that I'm taking below- I just want to state that outright- There are probably features of the pd.date_range method that I am not aware of yet that may take the problem I am faced with (leap year basically causing the freq to leave the first several observations of year 2021 in 2020, i.e. the observations for the bridges with STRUCNUM = 000000000000008 or 000000000000019 for example which are the first 2 bridges observed each year because the bridges are in ascending numerical order- are corresponding to dates like 2020-12-30 00:00:00.0000 and 2020-12-31 00:00:00.0000) The use of the DateOffset or dt.is_leap_year to deal with this problem would be less work and more efficient but I'm trying to get this application up and working at this point now just shy of a year since my Career Lab!




# 2016 1st bridge = 01 0002, last bridge = 58 0344R

# 2016
# Make Numpy arange as a datetime to make a range of dates for 2016

# MVP II:
# loop to make the dates


#def createdates (abmt_rc_yr, ): 
    
# not sure if this is the approach for making all the abmt_rc_dates_.... but the idea I have is to make the numbers associated with each filename_16, ... filename_22 into a list or dict and then say
"""for filename_16 thru filename_22 abmt_rc_dates_XXXX where the Xs are a stand-in for the different years-abmt_rc_dates_XXXX gets the correct year added to the end of the which is then set equal to np.arange(datetime(XXXX,1,1), datetime(XXXX,12,31), timedelta(hours=(525600/60)/filename_XX XX is a two digit year)) 
well, I'm onto something here- need to make the variable name into an iterable that can have its string manipulated by adding the four digit year onto the end of it and then dividing out the number of the observations into the number of hours between the observations of the different bridges which will then produce the seven sets of dates for each year which can then be made into the different date ranges for each year that can then be made into the datetime object that is made to be set as an axis on the abmt_rc_XXXX 7 times (for the seven different abmt_rc_XXXX slices of the overall abmt_rc variable) )"""
#abmt_rc_dates = []

#for 

# 
abmt_rc_dates_2016 = np.arange(datetime(2016,1,1), datetime(2016,12,31), timedelta(hours=4.364723467862481)).astype(datetime)


# CS1 hours=1.366401497426299
# CS2 4.364723467862481

# remove the very last entry from abmt_rc_dates_2016 (np.arange includes endpoint of the intervals, that or I haven't found the setting that allows me to set stop with hours minutes and seconds in the arguments)
*abmt_rc_dates_2016,_ = abmt_rc_dates_2016

# abmt_rc_dates_2016 returns a list- convert to an array:

abmt_rc_dates_2016 = np.asarray(abmt_rc_dates_2016)

# df.set_axis(ind, inplace=False) set the index
abmt_rc_2016 = abmt_rc_2016.set_axis(abmt_rc_dates_2016, inplace=False)

# need to run an np.polyfit here, and for each year to look for a curve of some sort, try up to 4 degree polynomials.


# Plot the data, just to see what it looks like


# df.index.to_pydatetime() 
# OR df['date'] = df['timestamp'].apply(timestamp_to_datetime) 
#abmt_rc_2016.index = abmt_rc_2016.index.to_pydatetime()


plt.scatter(abmt_rc_2016.index, abmt_rc_2016.CS1)
plt.title('Reinforced Concrete abutments (abmt_rc) 2016')
plt.xlabel('Date')
plt.ylabel('CS1')


# df['date_ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())



#ax = plt.gca()
#xticks = ax.get_xticks()
#xticks_dates = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in xticks]
#ax.set_xticklabels(xticks_dates)

# tips = sns.load_dataset("tips")
#sns.regplot(x=abmt_rc_2016.index, y="CS1", data=abmt_rc_2016)

#check dtype of index in abmt_rc_2016
datatypes = abmt_rc_2016.dtypes.index

# a, b = np.polyfit(date.astype(np.int64), ser2, 1)

#m2016, b2016 = np.polyfit(abmt_rc_2016.index.astype(np.int64), abmt_rc_2016.CS1, 1)

#Y = -1.0900525194435487e-18 * x + 2.543149493138786

#plt.scatter(abmt_rc_2016.index, abmt_rc_2016.CS1)

""" Now make the datetime64 conversion? """

# 2017
abmt_rc_dates_2017 = np.arange(datetime(2017,1,1), datetime(2018,1,1), timedelta(hours=4.097287184284378)).astype(datetime)

# abmt_rc_dates_2017 returns a list- convert to an array:

abmt_rc_dates_2017 = np.asarray(abmt_rc_dates_2017)

# CS1 hours=1.364698551176196
# CS2 4.097287184284378

# remove last entry- once this is done the variable becomes a list not an array of object anymore!
*abmt_rc_dates_2017,_ = abmt_rc_dates_2017

abmt_rc_2017 = abmt_rc_2017.set_axis(abmt_rc_dates_2017, inplace=False)

plt.scatter(abmt_rc_2017.index, abmt_rc_2017.CS1)

# 2018 1st bridge = 7540, last bridge = 10053

# 2018

abmt_rc_dates_2018 = np.arange(datetime(2018,1,1), datetime(2019,1,1), timedelta(hours=3.785652549697494)).astype(datetime)

# CS1 hours=1.36003726129483
# CS2 3.785652549697494

# remove last entry
# *abmt_rc_dates_2018,_ = abmt_rc_dates_2018

abmt_rc_2018 = abmt_rc_2018.set_axis(abmt_rc_dates_2018, inplace=False)

plt.scatter(abmt_rc_2018.index, abmt_rc_2018.CS1)

# 2019 1st bridge = 00008, last bridge = 10053

# 2019

abmt_rc_dates_2019 = np.arange(datetime(2019,1,1), datetime(2020,1,1), timedelta(hours=3.663739021329987)).astype(datetime)

# CS1 hours=1.359404096834264
# CS2 3.663739021329987


# remove last entry
# abmt_rc_dates_2019,_ = abmt_rc_dates_2019

abmt_rc_2019 = abmt_rc_2019.set_axis(abmt_rc_dates_2019, inplace=False)

plt.scatter(abmt_rc_2019.index, abmt_rc_2019.CS1)




# 2020 1st bridge = 4825, last bridge = 10053

# 2020

abmt_rc_dates_2020 = np.arange(datetime(2020,1,1), datetime(2020,12,31), timedelta(hours=3.218221895664952)).astype(datetime)

# abmt_rc_dates_2020 returns a list- convert to an array:

abmt_rc_dates_2020 = np.asarray(abmt_rc_dates_2020)

# CS1 hours=1.358139534883721
# CS2 3.218221895664952

# remove last entry
*abmt_rc_dates_2020,_ = abmt_rc_dates_2020

abmt_rc_2020 = abmt_rc_2020.set_axis(abmt_rc_dates_2020, inplace=False)

plt.scatter(abmt_rc_2020.index, abmt_rc_2020.CS1)




# 2021 1st bridge = 4825, last bridge = 10053

# 2021
# 09/11/2022 this is where the length mismatch of expected 6451 elements vs. 6450 occurs- could be that the length of each period needs to be recomputed!!!
# !!!
abmt_rc_dates_2021 = np.arange(datetime(2021,1,1), datetime(2022,1,1), timedelta(hours=3.095406360424028)).astype(datetime)

# CS1 hours=1.357929003255309
# CS2 3.095406360424028

# remove last entry
# *abmt_rc_dates_2021,_ = abmt_rc_dates_2021

abmt_rc_2021 = abmt_rc_2021.set_axis(abmt_rc_dates_2021, inplace=False)

plt.scatter(abmt_rc_2021.index, abmt_rc_2021.CS1)
# !!!
# 2022 1st bridge = 4825, last bridge = 10053

# 2022

abmt_rc_dates_2022 = np.arange(datetime(2022,1,1), datetime(2023,1,1), timedelta(hours=3.004115226337449)).astype(datetime)

# CS1 hours=1.358771521637971
# CS2 3.004115226337449


# remove last entry
#*abmt_rc_dates_2022,_ = abmt_rc_dates_2022

abmt_rc_2022 = abmt_rc_2022.set_axis(abmt_rc_dates_2022, inplace=False)

plt.scatter(abmt_rc_2022.index, abmt_rc_2022.CS1)

# 08/31/2022 stopped here.

# Create a the dataframe from the 7 abmt_rc_2016 - abmt_rc_2022 created above.  It will be called df_OLS_deck_rc to represent that the dataframe will then be used to carry out an ordinary least squares (OLS) regression analysis of this data.   

# Concatenate the dfs

df_regr_abmt_rc =pd.concat([abmt_rc_2016, abmt_rc_2017, abmt_rc_2018, abmt_rc_2019, abmt_rc_2020, abmt_rc_2021, abmt_rc_2022,], axis=0)

# Try going back to the ascending treatment of the CS and the apply polyfit and see if something that makes sense can be found- i.e. if there is a discernable uptick in the quantities of each CS over time.  

# e.g. deck_rc_2017 = deck_rc_2017.sort_values(by=['CS1'], ascending=True) 


    


# use iloc not loc
time_s = df_regr_abmt_rc.iloc[: ,0]
# Don't know if this is necessary

# move the timestamp column into the DataFrame to continue to use as part of the data.
df_regr_abmt_rc = df_regr_abmt_rc.reset_index(drop=False)

df_regr_abmt_rc = df_regr_abmt_rc.rename({'index': 'time'}, axis = 1)


df_regr_abmt_rc.reset_index(inplace=True)

df_regr_abmt_rc['time']=df_regr_abmt_rc['time'].apply(mpl_dates.date2num)
df_regr_abmt_rc['time'] = df_regr_abmt_rc['time'].astype(float)


# Start here 03/07/2023 with curve fitting and possibly just using a linear best fit package of some sort.  



# np.polyfit is probably not the solution to our problem as it requires that the equation be a polynomial- and we cannot see a curve that would seem to fit our data- so we need to go back to a linear model or a feature engineering attempt to make the model "more" linear, and better suited to an OLS model.

# Also the dates need to be put back to the state that will show as dates on the plots.

# go to https://stackoverflow.com/questions/56657323/typeerror-ufunc-add-cannot-use-operands-with-types-dtypem8ns-and-dtype for details to fix error.
x = np.asarray(df_regr_abmt_rc['time'])
y = np.asarray(df_regr_abmt_rc['CS1'])

curve = np.polyfit(x, y, 2)
print(curve)
poly = np.poly1d(curve)
print(poly)

plt.plot(x, y)
plt.show()

# with deg = 1, 2.791e-05 x + 0.3095
# with deg = 2, -7.835e-09 x + 0.0003117 x - 2.257


# abmt_rc_dates_2016 = np.asarray(abmt_rc_dates_2016)


# df_regr_abmt_rc.index.name='time'


# Stopped for the night here 11/27/22, trying to rename the index column
# df_regr_abmt_rc.rename({'index':'time', 'filename':'filename', 'STRUCNUM':'STRUCNUM', 'EN':'EN', 'TOTALQTY':'TOTALQTY', 'CS1':'CS1', 'CS2':'CS2', 'CS3':'CS3', 'CS4':'CS4'}, axis = 1, inplace=True)

# df_regr_abmt_rc.rename({'index': 'time'}, axis = 1, inplace=True)

# df_regr_abmt_rc = [['filename', 'STRUCNUM', 'EN', 'TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4', 'index']]


# end_cols = ['index']


# filename | STRUCNUM |EN | TOTALQTY | CS1 | CS2 | CS3 | CS4 

# df_regr_abmt_rc = [[c for c in df_regr_abmt_rc if c not in end_cols] + [c for c in end_cols if c in df_regr_abmt_rc]]




# Begin Regression of the concatenated set called df_regr_abmt_rc:

#sns.scatterplot(x='date_ordinal',y='CS2',data=df_regr_abmt_rc,color='darkorange')    


#https://stackoverflow.com/questions/24588437/convert-date-to-float-for-linear-regression-on-pandas-data-frame

# df = pd.read_csv('test.csv')
# df['date'] = pd.to_datetime(df['date'])    
# df['date_delta'] = (df['date'] - df['date'].min())  /  np.timedelta64(1,'D')
#city_data = df[df['city'] == 'London']
#result = sm.ols(formula = 'sales ~ date_delta', data = city_data).fit()


sns.pairplot(df_regr_abmt_rc, x_vars=['time'], y_vars = 'CS1', height = 7, aspect = 0.7, kind = 'reg')




# VERY IMPORTANT:
points = plt.scatter(x = df_regr_abmt_rc['date_ordinal'], y = df_regr_abmt_rc['CS1'], c=df_regr_abmt_rc['CS1'], s=75, cmap='BrBG')
plt.colorbar(points)
sns.regplot(x = df_regr_abmt_rc['date_ordinal'], y = df_regr_abmt_rc['CS1'], data=df_regr_abmt_rc, scatter=False, color='r')
ax = plt.gca()
xticks = ax.get_xticks()
fxticks_dates = [dt.date_ordinal.fromtimestamp(x).strftime('%Y-%m-%d') for x in xticks]
ax.set_xticklabels(xticks_dates)
plt.xticks(rotation=45)
#VERY IMPORTANT kwargs keyword arguments




# !!!
# TypeError: uns   upported operand type(s) for *: 'Timestamp' and 'float' 
df_regr_abmt_rc['date_ordinal'] = pd.to_datetime(df_regr_abmt_rc.index.to_series()).apply(lambda dt: dt.toordinal())

sns.scatterplot(
    data=df_regr_abmt_rc.reset_index(),
    y='CS2',
    x='index',
    color = 'darkorange')

ax = sns.regplot(
    data=df_regr_abmt_rc.reset_index(),
    x='index',
    y='CS2',
    scatter_kws={"color": "darkorange"}, line_kws={"color": "red"}
)
# Make the plot readable
ax.set_xlim(df_regr_abmt_rc['date_ordinal'].min() - 100, df_regr_abmt_rc['date_ordinal'].max() + 100)
ax.set_ylim(0, df_regr_abmt_rc['CS1'].max() + .25)

ax.set_xlabel('Date')
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)
plt.xticks(rotation=45)

# !!!
# Look out here: Make residuals plot

# draw residplot
sns.residplot(data=df_regr_abmt_rc,
x='date_ordinal',
y='CS1')

ax.set_xlabel('Date')
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)
plt.xticks(rotation=45)
#!!!

# Good to the line above!! (10/09/2022)

abmt_rc = getattr(element_df, '215', None)




# Try to replace missing data from years missing ENs covered in other years
# 

"""sns.lmplot(
    data=df_regr_abmt_rc,
    x='date_ordinal',
    y='CS1',
    scatter_kws={"color": "blue"}, line_kws={"color": "red"})


seaborn.residplot(data=None, *, x=None, y=None, x_partial=None, y_partial=None, lowess=False, order=1, robust=False, dropna=True, label=None, color=None, scatter_kws=None, line_kws=None, ax=None)
"""
    

# ax = df1.plot()
# df2.plot(ax=ax)

fig, ax = plt.subplots()



plt.title('Reinforced Concrete abutments (abmt_rc) 2016 to 2022, California')
plt.xlabel('Date')
plt.ylabel('CS1')
ax = plt.scatter(df_regr_abmt_rc.index, df_regr_abmt_rc.CS1)


slope, intercept = np.polyfit(df_regr_abmt_rc.index.astype(np.int64), df_regr_abmt_rc.CS1, 1)



# slope = -7.84080793412465e-20
# intercept = 1.0601339128916805



#fig, ax = plt.subplots()
# plot the line 
#a = pd.DataFrame({'a': [3,2,6,4]}, index = pd.date_range(dt(2019,1,1), periods = 4))
#plot = plt.plot_date(x=a.reset_index()['index'], y=a['a'], fmt="-")

# try to add the scatterplot
#b = pd.DataFrame({'b': [5, 2]}, index = pd.date_range(dt(2019,1,1), periods = 2))
#plot = plt.scatter(x=b.reset_index()['index'], y=b['b'], c='r')
#plt.show()


# fig, ax = plt.subplots()

# plot = plt.plot_date(x=a.reset_index()['index'], y=a['a'], fmt="-")

# b = pd.DataFrame({'b': [5, 2]}, index = pd.date_range(dt(2019,1,1), periods = 2))
# plot = plt.scatter(x=b.reset_index()['index'], y=b['b'], c='r')
# plt.show()


abline_values = [slope * i + intercept for i in df_regr_abmt_rc.index]

plt.plot(df_regr_abmt_rc.index.astype(np.int64), df_regr_abmt_rc.CS1, '--')
plt.plot(df_regr_abmt_rc.index.astype(np.int64), abline_values, 'b')
plt.title(slope)
plt.show() 

ax.axline((0, -7.84081e-20), slope=1.06013, color='C0', label='best fit')
# ax.set_xlim(0, 1)
ax.set_ylim(0, 1) 
ax.legend()






# Switching bridge elements here


# Remove the entries in the df for CS1 that are 0 and have already progressed through CS1 in entirety (meaning the element will not experience CS1 again unless an outside influence is applied to it such as replacement or repair).  



topFlg_rc = topFlg_rc.loc[~((topFlg_rc['CS1'] == 0.0) & (topFlg_rc['CS2'] + topFlg_rc['CS3'] + topFlg_rc['CS4'] == 1.0)),:]

topFlg_rc.reset_index(inplace = False)

filenames = topFlg_rc['filename'].tolist()

counts = {}

for file in filenames:
    if file in counts:
        counts[file] += 1
    else:
        counts[file] = 1

# the number of occurrences is placed below each counts.get: 
filename_16 = counts.get('2016CA_ElementData.xml')
# 3993

filename_17 = counts.get('2017CA_ElementData.xml')
# 4072

filename_18 = counts.get('2018CA_ElementData.xml')
# 4221

filename_19 = counts.get('2019CA_ElementData.xml')
# 4343

filename_20 = counts.get('2020CA_ElementData.xml')
# 4362

filename_21 = counts.get('2021CA_ElementData.xml')
# 4366

filename_22 = counts.get('2022CA_ElementData.xml')
# 4397

# for col in topFlg_rc.columns:    
  #        print(f"{col} has",topFlg_rc[col].value_counts()['2016CA_ElementData.xml']," 2016CA_ElementData.xml in it")
          
# topFlg_rc['filename'].value_counts()['2022CA_ElementData.xml']

# 2016 3993
# 2017 4072          
# 2018 4221
# 2019 4343
# 2020 4362
# 2021 4366
# 2022 4397

# Let's check out some other possible regression candidates:

# First we'll try top flange of reinforced concrete bridge girders, EN = 16

# 3993 periods/year
topFlg_rc_2016 = topFlg_rc.iloc[0:3993, :]

# abmt_rc_2016 = abmt_rc.iloc[0:6411, :]

# 4072
topFlg_rc_2017 = topFlg_rc.iloc[3993:8065, :]


# 4221
topFlg_rc_2018 = topFlg_rc.iloc[8065:12286, :]


# 4343
topFlg_rc_2019 = topFlg_rc.iloc[12286:16629, :]


# 4362
topFlg_rc_2020 = topFlg_rc.iloc[16629:20991, :]


# 4366
topFlg_rc_2021 = topFlg_rc.iloc[20991:25357, :]


# 4397
topFlg_rc_2022 = topFlg_rc.iloc[25357:29754, :]


# topFlg_rc dates
# 2016 3993 periods per year 

#abmt_rc_dates_2016 = np.arange(datetime(2016,1,1), datetime(2016,12,31), timedelta(hours=1.366401497426299)).astype(datetime)

topFlg_rc_dates_2016 = np.arange(datetime(2016,1,1), datetime(2016,12,31), timedelta(hours=2.193839218632607)).astype(datetime)

# remove the very last entry from deck_rc_dates_2016 (np.arange includes endpoint of the intervals, that or I haven't found the setting that allows me to set stop with hours minutes and seconds in the arguments)
*topFlg_rc_dates_2016,_ = topFlg_rc_dates_2016

# df.set_axis(ind, inplace=False) set the index
topFlg_rc_2016 = topFlg_rc_2016.set_axis(topFlg_rc_dates_2016, inplace=False)

# Plot the data, just to see what it looks like
plt.scatter(topFlg_rc_2016.index, topFlg_rc_2016.CS1)

# 2016 3993
# 2017 4072          
# 2018 4221
# 2019 4343
# 2020 4362
# 2021 4366
# 2022 4397

# 2017
topFlg_rc_dates_2017 = np.arange(datetime(2017,1,1), datetime(2018,1,1), timedelta(hours=2.151277013752456)).astype(datetime)

# remove last entry
# *topFlg_rc_dates_2017,_ = topFlg_rc_dates_2017

topFlg_rc_2017 = topFlg_rc_2017.set_axis(topFlg_rc_dates_2017, inplace=False)

plt.scatter(topFlg_rc_2017.index, topFlg_rc_2017.CS1)

# 2018 1st bridge = 7540, last bridge = 10053

# 2018

topFlg_rc_dates_2018 = np.arange(datetime(2018,1,1), datetime(2019,1,1), timedelta(hours=2.075337597725657)).astype(datetime)

# remove last entry
# *topFlg_rc_dates_2018,_ = topFlg_rc_dates_2018

topFlg_rc_2018 = topFlg_rc_2018.set_axis(topFlg_rc_dates_2018, inplace=False)

plt.scatter(topFlg_rc_2018.index, topFlg_rc_2018.CS1)

# 2019 1st bridge = 00008, last bridge = 10053

# 2019

topFlg_rc_dates_2019 = np.arange(datetime(2019,1,1), datetime(2020,1,1), timedelta(hours=2.017038913193645)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2019,_ = topFlg_rc_dates_2019

topFlg_rc_2019 = topFlg_rc_2019.set_axis(topFlg_rc_dates_2019, inplace=False)

# plt.scatter(topFlg_rc_2019.index, topFlg_rc_2019.CS1)




# 2020 1st bridge = 4825, last bridge = 10053

# 2020

topFlg_rc_dates_2020 = np.arange(datetime(2020,1,1), datetime(2020,12,31), timedelta(hours=2.008253094910591)).astype(datetime)

# remove last entry
# *topFlg_rc_dates_2020,_ = topFlg_rc_dates_2020

topFlg_rc_2020 = topFlg_rc_2020.set_axis(topFlg_rc_dates_2020, inplace=False)

# plt.scatter(topFlg_rc_2020.index, topFlg_rc_2020.CS1)


# 2021

topFlg_rc_dates_2021 = np.arange(datetime(2021,1,1), datetime(2022,1,1), timedelta(hours=2.006413192853871)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2021,_ = topFlg_rc_dates_2021

topFlg_rc_2021 = topFlg_rc_2021.set_axis(topFlg_rc_dates_2021, inplace=False)

# plt.scatter(topFlg_rc_2021.index, topFlg_rc_2021.CS1)


# 2022

topFlg_rc_dates_2022 = np.arange(datetime(2022,1,1), datetime(2023,1,1), timedelta(hours=1.992267455083011)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2022,_ = topFlg_rc_dates_2022

topFlg_rc_2022 = topFlg_rc_2022.set_axis(topFlg_rc_dates_2022, inplace=False)

# plt.scatter(topFlg_rc_2022.index, topFlg_rc_2022.CS1)




# datetime objects cannot be used as numeric value
# Convert the datetime object to a numeric value, perform the regression, 
# Plot the data

# The solution was brute force but the result is a df that ends each year as I orginally intended- which is that the last observation made in the years 2016 and 2020 would occur on December 31st of that year, and in the other years it occurs on January 1st of the following year.  

# The format of the 'Date' in the resulting df_OLS_deck_rc dataframe is in the form of '%Y-%m-%d %H:%M:%S.%f' meaning Year-Month-Day Hour:Minute:Second.Fraction.  

# Check data type of the "Date" in the dataframe

df_regr_topFlg_rc =pd.concat([topFlg_rc_2016, topFlg_rc_2017, topFlg_rc_2018, topFlg_rc_2019, topFlg_rc_2020, topFlg_rc_2021, topFlg_rc_2022,], axis=0) 


df_regr_topFlg_rc['date_ordinal'] = pd.to_datetime(df_regr_topFlg_rc.index.to_series()).apply(lambda dt: dt.toordinal())

ax = sns.regplot(
    data=df_regr_topFlg_rc,
    x='date_ordinal',
    y='CS1',
    scatter_kws={"color": "blue"}, line_kws={"color": "red"}
)
# Make the plot readable
ax.set_xlim(df_regr_topFlg_rc['date_ordinal'].min() - 100, df_regr_topFlg_rc['date_ordinal'].max() + 100)
ax.set_ylim(0, df_regr_topFlg_rc['CS1'].max() + .25)

ax.set_xlabel('Date')
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)
plt.xticks(rotation=45)


# !!!
# Look out here: Make residuals plot

# draw residplot
sns.residplot(data=df_regr_topFlg_rc,
x='date_ordinal',
y='CS1')



# Needs to have the xticks sorted out possibly- make the dates appear correctly
ax.set_xlabel('Date')
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)
plt.xticks(rotation=45)
df_regr_topFlg_rc.index.dtype

# result:
# Out[2]: dtype('<M8[ns]')


# BREAK!!!
# Plot the data
plt.scatter(df_regr_abmt_rc.index, df_regr_abmt_rc.CS1)

# tick_spacing = 5

# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#set variables need to be in specific format 
# X1 = df_OLS_deck_rc.DateTime.values.reshape(-1,1)
# x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]

# exec( i + " = df[i].values" )



X1 = df_regr_abmt_rc.index.values.reshape(-1,1)
y1 = df_regr_abmt_rc.CS1.values.reshape(-1,1)

#create train / test split for validation 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=0)


# plt.plot(data.index, data.price)


# fig, ax = plt.subplots(1,1)
# ax.plot(X_train1,y_train1)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# plt.show()

reg = LinearRegression().fit(X_train1, y_train1)
# Line betweent the check marks is where the problem starts! ("Could not be promoted by float64, no common DType exists for the given inputs, etc, etc.)
# !!!
reg.score(X_train1, y_train1)
# !!!

reg.coef_
y_hat1 = reg.predict(X_train1)

plt.scatter(X_train1,y_train1)
plt.scatter(X_train1,y_hat1)
plt.show()

y_hat_test1 = reg.predict(X_test1)
plt.scatter(X_test1, y_test1)
plt.scatter(X_test1, y_hat_test1)
plt.show()

# 08/20/22 DateTime key error persists- probably the column heading needs changing!  
# 08/21/22 

#MSE & RMSE penalize large errors more than MAE 
mae = mean_absolute_error(y_hat_test1,y_test1)
rmse = math.sqrt(mean_squared_error(y_hat_test1,y_test1))
print('Root Mean Squared Error = ',rmse)
print('Mean Absolute Error = ',mae)

import statsmodels.api as sm

X1b = df_regr_abmt_rc[['constant','DateTime']]
y1b = df_regr_abmt_rc.CS1.values

X_train1b, X_test1b, y_train1b, y_test1b = train_test_split(X1b, y1b, test_size=0.3, random_state=0)

reg_sm1b = sm.OLS(y_train1b, X_train1b).fit()
reg_sm1b.summary()

