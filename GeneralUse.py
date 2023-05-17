###### General Use Tools ######
#                             #
#      By: Colin Yancey       #
#      Created Feb. 2022      #
#    Last Edited: 7/7/2022    #
#        Schulman Lab         #
#  Johns Hopkins University   #
#                             #
###############################

import numpy as np

# Sets a floor and ceiling for a number
# num: Value to be buffered
# minVal: The returned value's lower bound
# maxVal: The returned value's upper bound
def bufferNumber(num, minVal=0, maxVal=1):
    return max(min(num, maxVal), minVal)

# Rounds to a specified count of significant figures
# x: Value to be rounded
# sigFig: Number of significant figures to round to
def roundToSigFig(x, sigFig=2):
    return 0 if x==0 else round(x, sigFig-int(np.floor(np.log10(abs(x))))-1)

# Returns a list with specific values from the original, specified by the requested indexes
# listObj: List of interest
# inds: Indexes of interest to pull from the list
def filterList(listObj, inds):
    return [x for i,x in enumerate(listObj) if i in inds]

# Sanitize an addendum to a dictionary's list entry that may or may not exist (checks for the existence of a list for the given key first)
# dictObj: The dictionary of interest
# key: The key for the given value
# addlList: The intended additional list to be appended
def safeAppend(listObj, key, addlList, rmInBetween=0):
    if key in listObj:
        listObj[key] = np.append(listObj[key][:(len(listObj[key]) - rmInBetween)], addlList)
    else:
        listObj[key] = addlList
