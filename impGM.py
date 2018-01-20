"""
Module for importing ground motion from a .csv file.
Input: filename [time step, acceleration]
Output: ground motion list
"""

import numpy as np
import pandas as pd
import csv

def impCsv(filename):
    # Read the CSV into a pandas data frame (df)
    #   With a df you can do many things
    #   most important: visualize data with Seaborn
    df = pd.read_csv(filename, delimiter=',', header=None)

    # Or export it in many ways, e.g. a list of tuples
    GM_a2 = df.values
    
    # Final time and time step
    timFin = GM_a2[-1][0]
    timSteGM = GM_a2[1][0]-GM_a2[0][0]
    
    return GM_a2,timFin,timSteGM
