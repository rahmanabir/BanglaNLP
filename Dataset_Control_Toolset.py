############
# @desc: 
#       A List of functions that will split/convert/change npy files,
#       Monitor loss deltas of a model that is being trained and change datasets when loss delta 
#       is stagnant. 


import pandas as pd
import numpy as np
import json




class ModelMonitor:

    # @desc: 
    #       Class that reads a list of datasets, 
    #       monitors loss of a model, 
    #       switches datasets when loss delta between epochs become stagnant    
    # @params:
    #       root_path <str>: path to where the datasets are kept
    #       dataset_names <list<str>>: the file names of said dataset
    #       max_stgnt <int>: the number of times loss delta can be low until dataset is changed 
    #       delta_val <float>: the delta value for which an epoch can be considered stagnant 
    #       logpath <str>: path+filename of where the log of the loss function will be stored
    # @example:
    #       root_path = 'datasets/'
    #       dataset_names = ['a.npy','b.npy','c.npy','d.npy']
    #       max_stgnt = 5
    #       delta_val = 0.001
    #       logpath = 'hello.json'
    #       logger = ModelMonitor(root_path, dataset_names, max_stgnt, delta_val, logpath)

    def __init__(self,  root_path, dataset_names, max_stgnt, delta_val, logpath='modelLogs.json'):
        
        self.root_path = root_path
        self.dataset_names = dataset_names
        self.max_stgnt = max_stgnt
        self.delta_val = delta_val

        self.current_loss = 0
        self.stagnant_counter = 0
        self.dataset_index = 0         
        self.current_dataset = dataset_names[0]


        self.loss_list = []
        self.logs = []
        self.logpath = logpath

    
    def save_logs(self):

        # @desc: saves logs in json format to a file

        with open(self.logpath, 'w') as outfile:
            json.dump({'logs' : self.logs}, outfile)



    
    def __switch(self):

        # @desc: checks loss situation and determines whether or not to switch datasets

        if(self.stagnant_counter >= self.max_stgnt):
            
            if(self.dataset_index+1 >= len(self.dataset_names) ):
                
                self.current_dataset =  -1 #signaling the end of training
    
                print('end of dataset reached logs will be saved in '+self.logpath)
                self.save_logs()
            
            else:

                logdict = {'filename' : self.dataset_names[self.dataset_index], 'loss' : self.loss_list}

                self.logs.append(logdict)

                self.dataset_index += 1 
                self.current_dataset =  self.dataset_names[self.dataset_index]


                        


    def update(self, loss):

        self.loss_list.append(loss)

        if(loss - self.current_loss < self.delta_val):
            self.stagnant_counter += 1

        self.current_loss = loss

        self.__switch()

