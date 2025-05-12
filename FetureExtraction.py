"""
This file contains a method that extract the features from the csv 
files contatining voltages in the channels and and event markers.
Author: Dan Shudrenko
"""

import os 
import numpy as np

FEATURES = os.path.join(".", "features")
os.makedirs(FEATURES, exist_ok=True)

def parse(path, n):
    """
    This function parse through the folder to find needed data
    """
    ##iterating through every subject
    file_names = os.listdir(path)
    for i in range(1, n+1, 1):
        #finding the files matching the index
        matches = []
        for name in file_names:
            if str(i).zfill(3) in name: 
                matches.append(name)
        #if the files are found 
        if len(matches):
            for u in range(2): 
                matches[u] = os.path.join(path, matches[u])
            events = np.genfromtxt(matches[0], delimiter= ",", skip_header=1, dtype = "float")
            data = np.genfromtxt(matches[1], delimiter= ",", skip_header=1, dtype = "float")
            if "events" in matches[1]:
                data, events = events, data
            featurize(data, events, i)
            
                
            

def featurize(data, events, n): 
    """
    This function extracts features from the two np arrays
    and saves it to the path defined by global variables FEATURES 
    """
    #modifying the data by getting rid of not needed channel
    chanel_interest = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11] #channel of interest
    data = data[:,chanel_interest]
    #going through the list of events 
    j = 0
    event_interest = [3, 5, 24, 40]
    result = []
    while j < events.shape[0]:
        #skipping the events that are withing bad block 
        while events[j][1] == 700000:
            bad_block_end = events[j][3]
            while j < events.shape[0] and events[j][2] <= bad_block_end:
                j += 1
            if j >= events.shape[0]:
                break
        if j >= events.shape[0]:
            break

        if events[j][1] in event_interest: 
            #ectracting data point  
            label = 1
            if events[j][1] == 3 or events[j][1] == 24:
                label = 0
            #checking if can add the next event
            event_start = int(events[j][2])
            event_end  = int(event_start + 640)
            i = j 
            while i < events.shape[0] and events[i][1] != 700000 and events[i][2] < event_end:
                i += 1
            #if we adding the event
            if i >= events.shape[0] or events[i][1] != 700000:
                instance = data[event_start:event_end, :]
                instance = instance.flatten("F")
                instance = np.insert(instance, 0, event_start)
                instance = np.insert(instance, 0, events[j][1])
                instance = np.insert(instance, 0, label)
                result.append(instance)
        j += 1
    res = np.array(result)
    file_path = os.path.join(FEATURES, "f_sub_" + str(n) + ".csv")
    np.savetxt(file_path, res, delimiter=',')




def main():
    parse("../subject_data/subject_data", 92)



if __name__ == "__main__":
    main()