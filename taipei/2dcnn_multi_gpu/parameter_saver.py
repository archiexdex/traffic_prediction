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

        self.config["index"] += 1
        self.index = int(self.config["index"])
        self.config[self.index] = {}
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
        self.config = {}
        self.file_name = "training_loss.json"
        try:
            with open(self.file_name) as fp:
                self.config = json.load(fp)

        except:
            print(">> " + self.file_name + " is not exist.")
            self.config["index"] = -1

        self.config["index"] += 1
        self.index = int(self.config["index"])    
        self.config[self.index] = {}
        self.config[self.index]["loss"] = []
        self.config[self.index]["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_parameter(self, name, value):
        """
        Param:
            name: the parameter name you want to store
            value: the value correspont to your name
        """
        self.config[self.index][name] = value

    def add_loss(self, step, train_loss, test_loss):
        """
        Param:
            train_loss: the training loss
            test_loss: the testing loss
        """
        self.config[self.index]["loss"].append([step, train_loss, test_loss])

    def save(self):
        """
        Param:
            
        """
        with open(self.file_name, 'w') as fp:
            json.dump(self.config, fp)

   
