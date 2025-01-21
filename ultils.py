import pandas as pd

def getRequiredFields(dataframe, required_fields):
    result = dataframe[required_fields]
    return result;