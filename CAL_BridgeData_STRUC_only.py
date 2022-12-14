# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:29:08 2021
@author: Chris
"""

import pandas as pd
import xml.etree.ElementTree as et

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import datetime as dt

import io

import os

from functools import reduce

import matplotlib.dates as mdates
from matplotlib.dates import date2num
import seaborn as sns
import time
import matplotlib.ticker as ticker

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

df2016=parse_XML("2016CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 56275 lines long

df2017=parse_XML("2017CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 75574 lines long

df2018=parse_XML("2018CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 78832 lines long

df2019=parse_XML("2019CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 82570 lines long

df2020=parse_XML("2020CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 83627 lines long

df2021=parse_XML("2021CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 83933 lines long

df2022=parse_XML("2022CA_ElementData.xml", ["FHWAED", "STATE", "STRUCNUM", "EN", "EPN", "TOTALQTY", "CS1", "CS2", "CS3", "CS4"])
# 85926 lines long

# Looking at how many occurrences of a single bridge occur in each year so that the total possible number of bridges being observed is accurate.  
df2016.groupby('STRUCNUM').count() 
""" 7578 """

df2017.groupby('STRUCNUM').count()
""" 10019 """

df2018.groupby('STRUCNUM').count()
""" 10439 """
    
df2019.groupby('STRUCNUM').count()
""" 10851 """

df2020.groupby('STRUCNUM').count()
""" 10873 """

df2021.groupby('STRUCNUM').count()
""" 10877 """

df2022.groupby('STRUCNUM').count()
""" 10899 """


""" the dfs were creating more entries than were present originally due to additional entries created when EN = EPN that also had entries for CS1-CS4 data as well """

df2016=df2016[df2016.EPN.isnull()]

# Drops the number of lines from 56275 to 48966

df2017=df2017[df2017.EPN.isnull()]
# Drops the number of lines from 75574 to 66481

df2018=df2018[df2018.EPN.isnull()]
# Drops the number of lines from 78832 to 66859

df2019=df2019[df2019.EPN.isnull()]
# Drops the number of lines from 82570 to 67318

df2020=df2020[df2020.EPN.isnull()]
# Drops the number of lines from 83627 to 67893

df2021=df2021[df2021.EPN.isnull()]
# Drops the number of lines from 83933 to 68184

df2022=df2022[df2022.EPN.isnull()]
# Drops the number of lines from 85926 to 68913


""" the ...EPN.isnull() expressions above are required to merge the data properly while resulting in a number of STRUCNUM smaller than 10899, that being the total number of possible matches between the datasets  """


# !!!

# MVP II is the Minimum Viable Product version II- or a program/data analysis that would supercede this one.  

# If I refer to "MVP II" and then comment out some code in that area I am referring to a functionality I have not achieved in this program and I would hope to make possible in a second version of this application were it to be updated.  

# !!!

# MVP II 
# Make the names of xml files be read into a column in the resulting dataframe as they are read from the files "automatically" rather than assigning the filename as I do below.  

# !!!

# MVP II- automate the steps below rather than inserting filename manually as is done here, but for now place the filename in a column in the dfs after parsing, filename is placed in the first column.

df2016.insert(0, 'filename_16', '2016CA_ElementData.xml')

df2017.insert(0, 'filename_17', '2017CA_ElementData.xml')

df2018.insert(0, 'filename_18', '2018CA_ElementData.xml')

df2019.insert(0, 'filename_19', '2019CA_ElementData.xml')

df2020.insert(0, 'filename_20', '2020CA_ElementData.xml')

df2021.insert(0, 'filename_21', '2021CA_ElementData.xml')

df2022.insert(0, 'filename_22', '2022CA_ElementData.xml')


# b_17 thru b_21 just mean "bridge number" aka STRUCNUM for the corresponding years- 2016 thru 2022, making a variable that holds the STRUCNUM as an array for each year as it would be right after being parsed into a dataframe.  In other words the b_17 - b_21 variables will be larger in size (i.e. no. of rows) than the dataframes as seen below once the STRUCNUM not present in all years are removed.  


b_16 = df2016['STRUCNUM'].to_numpy()

b_17 = df2017['STRUCNUM'].to_numpy()

b_18 = df2018['STRUCNUM'].to_numpy()

b_19 = df2019['STRUCNUM'].to_numpy()

b_20 = df2020['STRUCNUM'].to_numpy()

b_21 = df2021['STRUCNUM'].to_numpy()

b_22 = df2022['STRUCNUM'].to_numpy()


# !!! THIS IS SUBJECT TO CHANGE: The assumption I will use is that the data for all bridges (denoted by STRUCNUM) that are common to all the years of data being analyzed (2017 - 2021 in this case) is to be considered, meaning that the data (condition states of individual bridge elements) associated with the bridges common across 5 years shall be used even if the data provided for a bridge one year is not provided for all the other years or is provided sporadically for other years (i.e. if the bridge components (EN) rated for condition state one year are not rated for all years - BUT some components of said bridge are rated for all years being considered and analyzed then those condition states for those elements can be used as part of the data).  For instance, a very common bridge component (or EN, element number) is a deck constructed of reinforced concrete, which I refer to as deck_rc, and this EN is number 12 as denoted by the Federal Highway Administration (FHWA) Specification for the National Bridge Inventory, Bridge Elements.  As such it may be that the number of observations across the years considered is not the same for that elemeent number (EN) each year- some years may include condition states associated with that element in some but not all years, but it is my intention to use as many observations as possible for as many bridge parts as possible to attempt to make the computer model accurate.  Again, this is subject to change.  


# Determine the set of STRUCNUM common to all years observed.  

strucnum_in_all = list(set.intersection(*map(set, [b_16, b_17, b_18, b_19, b_20, b_21, b_22])))
# Sort the list so its contents will look more familiar to the user, i.e. be in numerical order.
strucnum_in_all = sorted(strucnum_in_all)
# Results in 9829 bridges starting with STRUCNUM = 01 0002 & ending with STRUCNUM = 58C0026.


# Remove STRUCNUM not present in all dfs

df2016 = df2016[np.isin(df2016['STRUCNUM'].to_numpy(), strucnum_in_all)]
# No. of lines 48966

df2017 = df2017[np.isin(df2017['STRUCNUM'].to_numpy(), strucnum_in_all)]
# No. of lines 66481

df2018 = df2018[np.isin(df2018['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 57830 lines long orig.
# No. of lines 66859

df2019 = df2019[np.isin(df2019['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 52036 lines long orig.
# No. of lines 67318

df2020 = df2020[np.isin(df2020['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 52619 lines long orig.
# No. of lines 67893

df2021 = df2021[np.isin(df2021['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 48562 lines long orig.
# No. of lines 68184

df2022 = df2022[np.isin(df2022['STRUCNUM'].to_numpy(), strucnum_in_all)]
# 48562 lines long orig.
# No. of lines 68913


# Then to run a few checks, I make the STRUCNUM into sets for each newly modified dataframe strucnum_2017_mod, etc.

strucnum_2016_mod = df2016['STRUCNUM'].unique()

strucnum_2017_mod = df2017['STRUCNUM'].unique()

strucnum_2018_mod = df2018['STRUCNUM'].unique()

strucnum_2019_mod = df2019['STRUCNUM'].unique()

strucnum_2020_mod = df2020['STRUCNUM'].unique()

strucnum_2021_mod = df2021['STRUCNUM'].unique()

strucnum_2022_mod = df2022['STRUCNUM'].unique()


# strucnum_2017_mod = strucnum_2018_mod = strucnum_2019_mod = strucnum_2021_mod = strucnum_2020_mod 



from array import *

strucnum_2016_mod.tolist()

strucnum_2017_mod.tolist()

strucnum_2018_mod.tolist()

strucnum_2019_mod.tolist()

strucnum_2020_mod.tolist()

strucnum_2021_mod.tolist()

strucnum_2022_mod.tolist()

import collections

# Is this the list of bridges common to all years that I would use to make the dfs that have the missing data I could replace?  

if collections.Counter(strucnum_2022_mod) == collections.Counter(strucnum_2021_mod):
    print ("The lists are identical")
else :
    print ("The lists are not identical")

# lists associated with df2022 and df2021 are identical.

# !!!
# qty_obs_2017 = {k : f'{v}_2017' for k, v in el_names.items()} ... Make the STRUCNUM of the list strucnum_in_all into individual variables (there will be 8847 of them I assume) using the code at the beginning of this line as a guide to come up with a variable that will have the nomenclature eN_in_sTrucnum_000000000000004 (the one shown here would be a variable for STRUCNUM = 000000000000004 without anything attached to it denoting the year assessed) as a means of storing the all EN associated with that STRUCNUM for each year so there will be 8847*5 = 44235 total of those variables 



# I think I need to scrap all of that and just find a method that will make the set of bridges that can merge on STRUCNUM and EN obvious and therefore not be forced tp come up with a means of removing individual or multiple excess lines of data from each year's individual df after coming up with a list of common EN per STRUCNUM across all years being assessed!!!!




# !!!

# Is this the place where I go and make all the necessary dfs for each EN PER YEAR and then follow that with code to add the time component for each individual year so as to avoid having to ensure the number of observations per year is the same?  THe answer to that question is yes.  

# Create EN dataframes for each year prior to concatenation of those dataframes to facilitate inserting the time component individually for each bridge element for that year based on number of observations made that year.  

# !!! Of course the comment above is making me wonder if the idea that I should at least be able to infer that the same amount of time has passed between successive observations of said bridge element (EN) may cause inaccuracy which would be the case if the observations of a part of a bridge were made over successive years (e.g. if STRUCNUM = 000000000000004 is observed in all years being considered but a condition state for EN = 12 IS NOT recorded each year) it would be difficult to say that the same amount of time has passed between each observation of the bridge element (EN).  

# !!! Going to change the protocol for this analysis as follows:

#  a.	IF a bridge (STRUCNUM) is not present in all years being considered that STRUCNUM (bridge) is eliminated from the data.
#       i. 	IF a bridge element present in a bridge not meeting the criteria outlined in a. above (i.e. a bridge or STRUCNUM that is not eliminated from the data because that STRUCNUM IS present in all years under consideration) is not present and its condition state observed and recorded in all years being considered that bridge element (EN) is eliminated from the data.  (So eliminate the bridges first if they aren???t present in all the years then eliminate the elements from the bridges if the elements are not present in all the years for those bridges)


# Now begins the merging of different dataframes: Going to start by merging the two longest which are df2021 and df2022

# The nomenclature for these merges is to place the two digit year corresponding to the next smallest df number at the beginning of the variable name as the merges are made, i.e. in the manner that df20_21_22 is merged below after the larger dfs of df2022 and df2021 have been merged.

df22_21 = pd.merge(df2022, df2021, suffixes=['_22', '_21'], on=['STRUCNUM'])
# pre merge the longest of the 2 dfs is 68913 lines long
# Post merge the length is 67758 lines long


# The next longest is df2020 at 67689 lines long

df20_22_21 = pd.merge(df2020, df22_21, on=['STRUCNUM'])
# pre merge the longest of the 2 dfs is df2021 at 67758 lines long
# Post merge the length of df20_22_21 is 67088 lines long


# Change of suffixes post merge

#df.rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)

df20_22_21.rename(columns={'FHWAED':'FHWAED_20', 'STATE':'STATE_20', 'EPN': 'EPN_20', 'TOTALQTY':'TOTALQTY_20', 'CS1':'CS1_20', 'CS2':'CS2_20', 'CS3':'CS3_20', 'CS4':'CS4_20',}, inplace = True)



# Of the remaining dfs df2019 is longest at 67116 lines long

df19_20_22_21 = pd.merge(df2019, df20_22_21, on=['STRUCNUM'])

# Post merge the length of df21_20_19_17 is 65899 lines long

# Change of suffixes post merge

df19_20_22_21.rename(columns={'FHWAED':'FHWAED_19', 'STATE':'STATE_19', 'EPN': 'EPN_19', 'TOTALQTY':'TOTALQTY_19', 'CS1':'CS1_19', 'CS2':'CS2_19', 'CS3':'CS3_19', 'CS4':'CS4_19',}, inplace = True)


# Merge of df2018 into the already merged years  2019 2020 2021 and 2022

df18_19_20_22_21 = pd.merge(df2018, df19_20_22_21, on=['STRUCNUM'])

# Post merge the length of df18_19_20_22_21 is 64628 lines long

# Change of suffixes post merge

df18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_18', 'STATE':'STATE_18', 'EPN': 'EPN_18', 'TOTALQTY':'TOTALQTY_18', 'CS1':'CS1_18', 'CS2':'CS2_18', 'CS3':'CS3_18', 'CS4':'CS4_18',}, inplace = True)

# Merge 2017 into the df
df17_18_19_20_22_21 = pd.merge(df2017, df18_19_20_22_21, on=['STRUCNUM'])

# Post merge the length of df18_19_20_22_21 is 63451 lines long

# Change of suffixes post merge

df17_18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_17', 'STATE':'STATE_17', 'EPN': 'EPN_17', 'TOTALQTY':'TOTALQTY_17', 'CS1':'CS1_17', 'CS2':'CS2_17', 'CS3':'CS3_17', 'CS4':'CS4_17',}, inplace = True)

# Merge 2016 into the df
df16_17_18_19_20_22_21 = pd.merge(df2016, df17_18_19_20_22_21, on=['STRUCNUM'])

# Post merge the length of df16_17_18_19_20_22_21 is 46204 lines long

# Change of suffixes post merge

df16_17_18_19_20_22_21.rename(columns={'FHWAED':'FHWAED_16', 'STATE':'STATE_16', 'EPN': 'EPN_16', 'TOTALQTY':'TOTALQTY_16', 'CS1':'CS1_16', 'CS2':'CS2_16', 'CS3':'CS3_16', 'CS4':'CS4_16',}, inplace = True)


# NEED TO CONVERT THE CS1 thru CS4s TO PERCENTAGES AS I DID BEFORE!!!!
""" Perhaps do that part after I've made all the new dfs.  """


# df16, df17, df18, df19, df20, df21 & df22 represent the subsets of df16_17_18_19_20_22_21 that will be removed from that dataframe to make individual dataframes to be concatenated later.  


# Select columns of the dataframe df16_17_18_19_20_22_21 to make the dataframe for each individual year:
    
# These datafames will be in the general form of the following column headings: filename | STRUCNUM | EN | TOTALQTY | CS1 | CS2 | CS3 | CS4 and will be specific to the year represented in the variable name dfXX where XX is the 2 digit year (in this case ranging from 17 to 21).

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


# Change the column headings of the first df:
df16.columns = ['filename', 'STRUCNUM', 'EN', 'TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']


# concatenate the dataframes to one another starting with year 2017
# The dataframe holding all the years in order will be called df_data

df_data =pd.DataFrame(np.concatenate([df16.values, df17.values, df18.values, df19.values, df20.values, df21.values, df22.values], axis=0), columns=df16.columns)

# !!!

# Convert the columns to numeric - admittedly the filename probably does not need to be numeric, but I'm trying to get this done.
df_data[['TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']] = df_data[['TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']].apply(pd.to_numeric, errors='coerce')


# Divide the CS1 thru CS4 by TOTALQTY to make the condition state into a percentage of the total element per bridge
df_data[['CS1','CS2','CS3','CS4']] = df_data[['CS1','CS2','CS3','CS4']].div(df_data.TOTALQTY, axis=0)



# el_names means Element Names

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
            	'103': 'cwBg_pc',
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
            '522': 'deck_memb'
    }

""" 06/07/22 """


# Make a dictionary of the keys and values of the bridge element numbers (EN) and the name I intend to give to the variable that will hold the number observations of that EN for that year using a dictionary comprehension.

#qty_obs_2017 = {k : f'{v}_2017' for k, v in el_names.items()}





# Create the means to make the individual dataframes for each Element Number (EN) using getattr().

class df_names:
    pass

element_df = df_names() # element_df will hold all the variables (which will also be variables in the form of dataframes) to be created associated with the different bridge elements or  ENs.  

# Get the unique set of all EN common to all years being obsesrved/inventoried.  
elements = df_data['EN'].unique()


for element in elements:
    
    setattr(element_df, f"{element}", df_data[df_data['EN']==element].reset_index(drop=True))
    
for element in elements: #very important to keep the two for loops indented to the same spot! Second loop is not a nested loop!
        print(getattr(element_df, f"{element}"))



# MVP II 
# dictionary of all the bridge elements possible in any bridge.  
# plan to use the dict below as a means to create dataframes for the element numbers (ENs) present in the concatenated dataframe using a loop if possible rather than just using getattr for all the numbers in the dict and possibly making empty dataframes in the process.
# Although doing this may require using glob (global variables) which from my investigation of the practice seems like a poor attempt at a solution.  

# !!! This may not be necessary given the syntax of the getattr() statement, can return the None object and eliminate the need to only create dataframes for the ENs that are present in the observations.  If NoneType object is returned the program can continue on to the next EN without throwing an error- there will be a variable of Type = NoneType in the output.  
# !!!

"""
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
            	'103': 'cwBg_pc',
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
            '522': 'deck_memb'
    }
"""


# Create dataframes for each individual EN to perform regression analysis for each possible part of a bridge

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
    
# Rationale for the replacement of data for deck_rc is that the subset of data will consist of all bridges that have observations in all years AND at least one EN observation in one year- thus replacing the the EN observations for years where no data is present but at least one observation is present in at least one year for a bridge.      

# XX of the possible EN are NoneType objects (i.e. there are no observations of those EN present across all years being considered) which is not surprising because the list of total possible elements is exhaustive and many of the total of  124 elements are specialized types of constructin that do not see use in most typical highway briidges.  


# Create time column for the plots.  
# Make the required dfs and then make plots and perform regression.  
# Create a column to compute the percentage of each CS (condition state) as means to determine when that CS might overtake all of the elements to which it pertains.  


# I expect the most commonly used elements present in the data to be the type with the suffix _rc meaning reinforced concrete and _pc meaning prestressed concrete.  

# !!!
# simple linear regression for deck_rc or reinforced concrete deck
# !!!

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# https://stackoverflow.com/questions/70949098/how-to-work-around-the-date-range-limit-in-pandas-for-plotting

# abmt_rc Bridge abutments, reinforced concrete

# Going to do some work to this dataframe before using it to make a regression model

# The abmt_rc df is 61376 lines long and has 7 years of data 61376/7 = 8768 entries per year.  


# Create a datetime object to use for the abmt_rc df:
# Note how the range is stopped at 12/30/2021 to make the number of intervals between observations the same for each year and to allow for the 2020 leap year.  00:00:00.000000

# iloc[row slicing, column slicing]

# slicing the deck_rc dataframe into its individual years to assign date_range accurately.
  
# 8768 periods/year

# Let's remove any CS1 entries in the abmt_rc dataframe that have progressed through CS1 entirely and will only produce subsequent CS1 observations if an outside influence acts upon that bridge element.   I take outside influence to generally mean replacement of the element completely which would mean the element was new and by default in a condition state of CS1- although there may be refurbishment or repair actions that may return the element to a state of CS1.  

# So originally after the abmt_rc entries were placed into their own dataframe the number of rows was 61376.  

abmt_rc = abmt_rc.loc[~((abmt_rc['CS1'] == 0.0) & (abmt_rc['CS2'] + abmt_rc['CS3'] + abmt_rc['CS4'] == 1.0)),:]

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

# count all the occurrences of the different filenames in the list and palce them in a dict called counts.
filenames = abmt_rc['filename'].tolist()

counts = {}

for file in filenames:
    if file in counts:
        counts[file] += 1
    else:
        counts[file] = 1

# the number of occurrences is placed below each counts.get: 
filename_16 = counts.get('2016CA_ElementData.xml')
# 6411

filename_17 = counts.get('2017CA_ElementData.xml')
# 6419

filename_18 = counts.get('2018CA_ElementData.xml')
# 6441

filename_19 = counts.get('2019CA_ElementData.xml')
# 6444

filename_20 = counts.get('2020CA_ElementData.xml')
# 6450

filename_21 = counts.get('2021CA_ElementData.xml')
# 6451

filename_22 = counts.get('2022CA_ElementData.xml')
# 6447

# Slicing out the different years from the dataframe for all 7 years:
abmt_rc_2016 = abmt_rc.iloc[0:6411, :]

# The line of code below is to make the order of the CS1 entries in ascending order so as to make the line of best fit slope upwards as I have hypothesized it would.  The rationale for this approach is to say that the bridges can be observed/inspected in the field in any order we wish

abmt_rc_2016 = abmt_rc_2016.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2017 = abmt_rc.iloc[6411:12830, :]

abmt_rc_2017 = abmt_rc_2017.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2018 = abmt_rc.iloc[12830:19271, :]

abmt_rc_2018 = abmt_rc_2018.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2019 = abmt_rc.iloc[19271:25715, :]

abmt_rc_2019 = abmt_rc_2019.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2020 = abmt_rc.iloc[25715:32165, :]

abmt_rc_2020 = abmt_rc_2020.sort_values(by=['CS1'], ascending=True) 


abmt_rc_2021 = abmt_rc.iloc[32165:38616, :]

abmt_rc_2021 = abmt_rc_2021.sort_values(by=['CS1'], ascending=True)


abmt_rc_2022 = abmt_rc.iloc[38616:45063, :]

abmt_rc_2022 = abmt_rc_2022.sort_values(by=['CS1'], ascending=True)


# Make the first year (2016) ordered from lowest CS1 to highest CS1 - NOT ordered from lowest to highest STRUCNUM as they are currently. 

# I'm not happy about this approach that I'm taking below- I just want to state that outright- There are probably features of the pd.date_range method that I am not aware of yet that may take the problem I am faced with (leap year basically causing the freq to leave the first several observations of year 2021 in 2020, i.e. the observations for the bridges with STRUCNUM = 000000000000008 or 000000000000019 for example which are the first 2 bridges observed each year because the bridges are in ascending numerical order- are corresponding to dates like 2020-12-30 00:00:00.0000 and 2020-12-31 00:00:00.0000) The use of the DateOffset or dt.is_leap_year to deal with this problem would be less work and more efficient but I'm trying to get this application up and working at this point now just shy of a year since my Career Lab!




# 2016 1st bridge = 01 0002, last bridge = 58 0344R

# 2016
# Make Numpy arange as a datetime to make a range of dates for 2016

# 
abmt_rc_dates_2016 = np.arange(datetime(2016,1,1), datetime(2016,12,31), timedelta(hours=1.366401497426299)).astype(datetime)

# remove the very last entry from deck_rc_dates_2017 (np.arange includes endpoint of the intervals, that or I haven't found the setting that allows me to set stop with hours minutes and seconds in the arguments)
# *abmt_rc_dates_2016,_ = abmt_rc_dates_2016

# df.set_axis(ind, inplace=False) set the index
abmt_rc_2016 = abmt_rc_2016.set_axis(abmt_rc_dates_2016, inplace=False)

# Plot the data, just to see what it looks like
plt.scatter(abmt_rc_2016.index, abmt_rc_2016.CS1)


""" Now make the datetime64 conversion? """

# 2017
abmt_rc_dates_2017 = np.arange(datetime(2017,1,1), datetime(2018,1,1), timedelta(hours=1.364698551176196)).astype(datetime)

# remove last entry
*abmt_rc_dates_2017,_ = abmt_rc_dates_2017

abmt_rc_2017 = abmt_rc_2017.set_axis(abmt_rc_dates_2017, inplace=False)

plt.scatter(abmt_rc_2017.index, abmt_rc_2017.CS1)

# 2018 1st bridge = 7540, last bridge = 10053

# 2018

abmt_rc_dates_2018 = np.arange(datetime(2018,1,1), datetime(2019,1,1), timedelta(hours=1.36003726129483)).astype(datetime)

# remove last entry
*abmt_rc_dates_2018,_ = abmt_rc_dates_2018

abmt_rc_2018 = abmt_rc_2018.set_axis(abmt_rc_dates_2018, inplace=False)

plt.scatter(abmt_rc_2018.index, abmt_rc_2018.CS1)

# 2019 1st bridge = 00008, last bridge = 10053

# 2019

abmt_rc_dates_2019 = np.arange(datetime(2019,1,1), datetime(2020,1,1), timedelta(minutes=1.359404096834264)).astype(datetime)

# remove last entry
*abmt_rc_dates_2019,_ = abmt_rc_dates_2019

abmt_rc_2019 = abmt_rc_2019.set_axis(abmt_rc_dates_2019, inplace=False)

plt.scatter(abmt_rc_2019.index, abmt_rc_2019.CS1)




# 2020 1st bridge = 4825, last bridge = 10053

# 2020

abmt_rc_dates_2020 = np.arange(datetime(2020,1,1), datetime(2020,12,31), timedelta(hours=1.358139534883721)).astype(datetime)

# remove last entry
*abmt_rc_dates_2020,_ = abmt_rc_dates_2020

abmt_rc_2020 = abmt_rc_2020.set_axis(abmt_rc_dates_2020, inplace=False)

plt.scatter(abmt_rc_2020.index, abmt_rc_2020.CS1)




# 2021 1st bridge = 4825, last bridge = 10053

# 2021

abmt_rc_dates_2021 = np.arange(datetime(2021,1,1), datetime(2022,1,1), timedelta(hours=1.358139534883721)).astype(datetime)

# remove last entry
*abmt_rc_dates_2021,_ = abmt_rc_dates_2021

abmt_rc_2021 = abmt_rc_2021.set_axis(abmt_rc_dates_2021, inplace=False)

plt.scatter(abmt_rc_2021.index, abmt_rc_2021.CS1)

# 2022 1st bridge = 4825, last bridge = 10053

# 2022

abmt_rc_dates_2022 = np.arange(datetime(2022,1,1), datetime(2023,1,1), timedelta(hours=1.358771521637971)).astype(datetime)

# remove last entry
*abmt_rc_dates_2022,_ = abmt_rc_dates_2022

abmt_rc_2022 = abmt_rc_2022.set_axis(abmt_rc_dates_2022, inplace=False)

plt.scatter(abmt_rc_2022.index, abmt_rc_2022.CS1)

# 08/31/2022 stopped here.

# Create a the dataframe from the 7 abmt_rc_2016 - abmt_rc_2022 created above.  It will be called df_OLS_deck_rc to represent that the dataframe will then be used to carry out and ordinary least squares (OLS) regression analysis of this data.   

df_regr_abmt_rc =pd.concat([abmt_rc_2016, abmt_rc_2017, abmt_rc_2018, abmt_rc_2019, abmt_rc_2020, abmt_rc_2021, abmt_rc_2022,], axis=0) 


# Switching bridge elements here


# Remove the entries in the df for CS1 that are 0 and have already progressed through CS1 in entirety (meaning the element will not experience CS1 again unless an outside influence is applied to it such as replacement or repair).  

topFlg_rc = topFlg_rc.loc[~((topFlg_rc['CS1'] == 0.0) & (topFlg_rc['CS2'] + topFlg_rc['CS3'] + topFlg_rc['CS4'] == 1.0)),:]

for col in topFlg_rc.columns:    
          print(f"{col} has",topFlg_rc[col].value_counts()['2016CA_ElementData.xml']," 2016CA_ElementData.xml in it")
          
topFlg_rc['filename'].value_counts()['2022CA_ElementData.xml']

# 2016 5600
# 2017 5600          
# 2018 5774
# 2019 5941
# 2020 5959
# 2021 5967
# 2022 5999

# Let's check out some other possible regression candidates:

# First we'll try top flange of reinforced concrete bridge girders, EN = 16

# 6150 periods/year
topFlg_rc_2016 = topFlg_rc.iloc[0:5600, :]

topFlg_rc_2016 = topFlg_rc_2016.sort_values(by=['CS1'], ascending=True) 

# +5600
topFlg_rc_2017 = topFlg_rc.iloc[5600:11200, :]

topFlg_rc_2017 = topFlg_rc_2017.sort_values(by=['CS1'], ascending=True) 

# +5774
topFlg_rc_2018 = topFlg_rc.iloc[11200:16974, :]

topFlg_rc_2018 = topFlg_rc_2018.sort_values(by=['CS1'], ascending=True) 

# +5941
topFlg_rc_2019 = topFlg_rc.iloc[16974:22915, :]

topFlg_rc_2019 = topFlg_rc_2019.sort_values(by=['CS1'], ascending=True) 

# +5959
topFlg_rc_2020 = topFlg_rc.iloc[22915:28874, :]

# Problem starts here, length mismatch, 5959 elements expected, new values have 5658 elements.  May also need to get the index reset in the topFlg_rc df post dropping of the CS1 == 0 has been accomplished.  

topFlg_rc_2020 = topFlg_rc_2020.sort_values(by=['CS1'], ascending=True) 

# +5967
topFlg_rc_2021 = topFlg_rc.iloc[28874:34841, :]

topFlg_rc_2021 = topFlg_rc_2021.sort_values(by=['CS1'], ascending=True)

# +5999
topFlg_rc_2022 = topFlg_rc.iloc[34841:40840, :]



topFlg_rc_2022 = topFlg_rc_2022.sort_values(by=['CS1'], ascending=True)

# topFlg_rc dates
# 6150 periods per year

topFlg_rc_dates_2016 = np.arange(datetime(2016,1,1), datetime(2016,12,31), timedelta(hours=1.564285714285714)).astype(datetime)

# remove the very last entry from deck_rc_dates_2017 (np.arange includes endpoint of the intervals, that or I haven't found the ssetting that allows me to set stop with hours minutes and seconds in the arguments)
*topFlg_rc_dates_2016,_ = topFlg_rc_dates_2016

# df.set_axis(ind, inplace=False) set the index
topFlg_rc_2016 = topFlg_rc_2016.set_axis(topFlg_rc_dates_2016, inplace=False)

# Plot the data, just to see what it looks like
plt.scatter(topFlg_rc_2016.index, topFlg_rc_2016.CS1)


""" Now make the datetime64 conversion? """

# 2017
topFlg_rc_dates_2017 = np.arange(datetime(2017,1,1), datetime(2018,1,1), timedelta(hours=1.564285714285714)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2017,_ = topFlg_rc_dates_2017

topFlg_rc_2017 = topFlg_rc_2017.set_axis(topFlg_rc_dates_2017, inplace=False)

plt.scatter(topFlg_rc_2017.index, topFlg_rc_2017.CS1)

# 2018 1st bridge = 7540, last bridge = 10053

# 2018

topFlg_rc_dates_2018 = np.arange(datetime(2018,1,1), datetime(2019,1,1), timedelta(hours=1.517145826117077)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2018,_ = topFlg_rc_dates_2018

topFlg_rc_2018 = topFlg_rc_2018.set_axis(topFlg_rc_dates_2018, inplace=False)

plt.scatter(topFlg_rc_2018.index, topFlg_rc_2018.CS1)

# 2019 1st bridge = 00008, last bridge = 10053

# 2019

topFlg_rc_dates_2019 = np.arange(datetime(2019,1,1), datetime(2020,1,1), timedelta(hours=1.474499242551759)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2019,_ = topFlg_rc_dates_2019

topFlg_rc_2019 = topFlg_rc_2019.set_axis(topFlg_rc_dates_2019, inplace=False)

plt.scatter(topFlg_rc_2019.index, topFlg_rc_2019.CS1)




# 2020 1st bridge = 4825, last bridge = 10053

# 2020

topFlg_rc_dates_2020 = np.arange(datetime(2020,1,1), datetime(2020,12,31), timedelta(hours=1.470045309615707)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2020,_ = topFlg_rc_dates_2020

topFlg_rc_2020 = topFlg_rc_2020.set_axis(topFlg_rc_dates_2020, inplace=False)

plt.scatter(topFlg_rc_2020.index, topFlg_rc_2020.CS1)




# 2021 1st bridge = 4825, last bridge = 10053

# 2021

topFlg_rc_dates_2021 = np.arange(datetime(2021,1,1), datetime(2022,1,1), timedelta(hours=1.46807440925088)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2021,_ = topFlg_rc_dates_2021

topFlg_rc_2021 = topFlg_rc_2021.set_axis(topFlg_rc_dates_2021, inplace=False)

plt.scatter(topFlg_rc_2021.index, topFlg_rc_2021.CS1)

# 2022 1st bridge = 4825, last bridge = 10053

# 2022

topFlg_rc_dates_2022 = np.arange(datetime(2022,1,1), datetime(2023,1,1), timedelta(hours=1.460243373895649)).astype(datetime)

# remove last entry
*topFlg_rc_dates_2022,_ = topFlg_rc_dates_2022

topFlg_rc_2022 = topFlg_rc_2022.set_axis(topFlg_rc_dates_2022, inplace=False)

plt.scatter(topFlg_rc_2022.index, topFlg_rc_2022.CS1)

# datetime objects cannot be used as numeric value
# Convert the datetime object to a numeric value, perform the regression, 
# Plot the data

# The solution was brute force but the result is a df that ends each year as I orginally intended- which is that the last observation made in the year 2020 would occur on December 30th of that year, and in the other 4 years it occurs on December 31st.  

# The format of the 'Date' in the resulting df_OLS_deck_rc dataframe is in the form of '%Y-%m-%d %H:%M:%S.%f' meaning Year-Month-Day Hour:Minute:Second.Fraction.  

# Check data type of the "Date" in the dataframe

df_regr_abmt_rc.index.dtype

# result:
# Out[2]: dtype('<M8[ns]')



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

