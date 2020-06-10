import os, sys 

from lib.config import TrainConfig

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        if not os.path.exists(TrainConfig().folder):
            os.mkdir(TrainConfig().folder)
        self.log = open(TrainConfig().folder + "/log.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass
