import os 
import sys



class Param:
    def __init__(self):
        ##############   Data Process   ##################
        self.code = '600019'
        self.stockDataPath = "./database/stockData"
        self.trainDatasetPath = "./database/trainDataset"
        self.testDatasetPath = "./database/testDataset"
        
        
        ##############  Neural Network  ################### 
        
        self.residualNumLSTM = 3
        self.residualNumFC = 3
        self.hiddenDim =256
        self.dropoutLSTM = 0.2
        
        ##############      Agent       ###################
        self.K_epochs = 20          # update policy for K epochs in one PPO update
        self.eps_clip = 0.2          # clip parameter for PPO
        self.gamma = 0.99            # discount factor
        self.lr_actor = 0.0001       # learning rate for actor network
        self.lr_critic = 0.0001      # learning rate for critic network
        
        ##############        Env       ###################
        
        self.maxStock = 150000
        self.RewardDay = 2
        self.fixed_quantity = False
        self.balanceStockInState = False
        self.seq_length=10
        
        self.balanceScaling = 10000000
        self.punishment = 1000
        
        self.initial_balance=100000
        self.initial_stock_owned=5000
        self.transaction_fee=1e-5
        ##############  train and test  ###################
        
        self.updateNum = 100          # after this num steps ,agent will update itself
        self.trainMode = True        # True == Train    False == Test
         
        self.episodeNum = 1          # train episode number
        self.stepNum = 8000          # step number in every episode 
        self.envName = self.setEnvName()
        self.log_dir = self.setLogDir()
        self.params_file_path = self.setParamsFilePath()
        self.modelSavePath = './model'
        self.resultPath = ''
        self.createDir()


        
    def createDir(self):
        if not os.path.exists(self.stockDataPath):
            os.makedirs(self.stockDataPath)
        if not os.path.exists(self.trainDatasetPath):
            os.makedirs(self.trainDatasetPath)
        if not os.path.exists(self.testDatasetPath):
            os.makedirs(self.testDatasetPath)
        if not os.path.exists(self.modelSavePath):
            os.makedirs(self.modelSavePath)
        
    def setEnvName(self):
        head = '_fixed_' if self.fixed_quantity else '_unfixed_'
        stateMode = "in" if self.balanceStockInState else "ex" 

        name = f'{self.seq_length}d{self.code}L{self.residualNumLSTM}F{self.residualNumFC}R{self.RewardDay}H{self.hiddenDim}'\
        + f'U{self.updateNum}K{self.K_epochs}DP{self.dropoutLSTM}' + head + stateMode
        return name
    
    def setLogDir(self):
        logDir = os.path.join("log", self.envName)
        return logDir
            
    def setParamsFilePath(self):
        filePath = os.path.join(self.log_dir, f"params.txt")
        return filePath
    


example = Param()
example.createDir()











