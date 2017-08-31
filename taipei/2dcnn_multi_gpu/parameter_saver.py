from datetime import datetime
import json
import time

class Parameter_saver(object):
    def __init__(self, algorithm_name):
        """
        Param:
            algorithm_name: the algorithm name
        """
        self.config = {}
        self.file_name = "training_config.json"
        try:
            with open(self.file_name) as fp:
                self.config = json.load(fp)

        except:
            print(">> " + self.file_name + " is not exist.")
            self.config["index"] = -1

        self.index = str(self.config["index"])
        if "flg" not in self.config[self.index]:            
            self.config["index"] += 1
            self.index = str(self.config["index"])
            self.config[self.index] = {}
            

        self.index = str(self.config["index"])

        self.config[self.index]["algorithm_name"] = algorithm_name
        self.config[self.index]["structure"] = []
        self.config[self.index]["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    
    def add_parameter(self, name, value):
        """
        Param:
            name: the parameter name you want to store
            value: the value correspont to your name
        """
        self.config[self.index][name] = value

    def save(self):
        """
        Param:
            
        """
        with open(self.file_name, 'w') as fp:
            json.dump(self.config, fp)

    def add_layer(self, name, kernal_size):
        """
        Param:
            name: the parameter name you want to store
            value: the value correspont to your name
        """
        self.config[self.index]["structure"].append([name, kernal_size])

class Training_loss_saver(object):
    def __init__(self):
        """
        Param:

        """
        self.path = "train_loss/"
        self.config = {}
        self.config_filename = "training_config.json"
        try:
            with open(self.config_filename) as fp:
                self.config = json.load(fp)

        except:
            print(">> " + self.config_filename + " is not exist.")
            self.config["index"] = -1

        self.config["index"] += 1
        self.index = self.config["index"]
        self.file_name = "train_loss_" + str(self.index) + ".json"
        
        self.config[self.index] = {}
        self.config[self.index]["flg"] = 1
        
        with open(self.config_filename, 'w') as fp:
            json.dump(self.config, fp)

        self.loss_list = {}
        self.loss_list["loss"] = []

    def add_parameter(self, name, value):
        """
        Param:
            name: the parameter name you want to store
            value: the value correspont to your name
        """
        self.config[self.index][name] = value
        with open(self.config_filename, 'w') as fp:
            json.dump(self.config, fp)

    def add_loss(self, step, train_loss, each_train_loss, test_loss, each_test_loss):
        """
        Param:
            train_loss: the training loss
            test_loss: the testing loss
        """
        # self.config[self.index]["loss"].append([step, train_loss, test_loss])
        self.loss_list["loss"].append( {"step"       : step, 
                                        "train_loss" : train_loss,
                                        "each_train_loss" : each_train_loss,
                                        "test_loss"  : test_loss,
                                        "each_test_loss" : each_test_loss})

    def save(self):
        """
        Param:
            
        """
        with open(self.path + self.file_name, 'w') as fp:
            json.dump(self.loss_list, fp)

   
