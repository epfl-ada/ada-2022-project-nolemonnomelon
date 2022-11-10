##################################################################
# This script contains the functions that are useful to 
# preprocess the data
##################################################################

def transform_into_dict(text) :
    ''' Transform the Freebase ID:name tuples strings into a dictionary for the genres abd countries '''
    if text != '{}' :
        text = text.replace('"','')[1:-1]
        return dict(subString.split(": ") for subString in text.split(", "))
    else :
        return float('NaN')
    
def transform_into_dict_2(text) :
    ''' Transform the Freebase ID:name tuples strings into a dictionary for the languages. This is a specific function for this feature only because
    there are , inside values of the dictionnary (Example : 'Thai, Northeastern Language' at index 1754) '''
    if text != '{}' :
        text = text[2:-2]
        text = text.replace(', "', '')
        text = text.replace(': ', '')
        return dict(subString.split('""') for subString in text.split('"/'))
    else :
        return float('NaN')
    
def transform_to_list_names(d) :
    ''' Transform a dict into a list of its values '''
    if isinstance(d, dict):
        return list(d.values())
    else :
        return d