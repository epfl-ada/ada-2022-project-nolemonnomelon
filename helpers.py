##################################################################
# This script contains some helper functions
##################################################################


import pandas as pd
import numpy as np

def transform_into_list(text) :
    ''' This function take a string and transform it into list of strings'''
    if isinstance(text, str) :
        if text != '[]' :
            text = text[1:-1]
            text = text.replace("'", "")
            return list(subString for subString in text.split(', '))
    else :
        return float('NaN')   
