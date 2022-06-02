from tensorflow import keras
from data.data_loader import DataLoader
from Logging.LoggerFile import Logger
import os

class ModelCreation:


    def __init__(self):
        # getting data
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = DataLoader().getValidationData()

        
        

    def CreateModel(self):


        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        try:
            Logger().log(f, 'Creating layers...')
            ## creating and connecting layers
            self.input_layer = keras.layers.Input(shape=self.X_train[1:])
            self.hidden1 = keras.layers.Dense(40, activation='relu')(self.input_layer)
            self.hidden2 = keras.layers.Dense(40, activation='relu')(self.hidden1)
            self.concat = keras.layers.Concatenate([self.input_layer, self.hidden2])
            self.output_layer = keras.layers.Dense(100, activation='softmax')

            Logger().log(f, 'Layers created successfully!')
            Logger().log(f, 'Creating a model...')
            ## creating a model
            self.model = keras.Model(inputs =[self.input_layer], outputs=[self.output_layer])
            Logger.log(f, 'Model created successfully!')
            f.close()
            return self.model

        except Exception as e:
            Logger().log(f, f"Exception occured while creating layers or creating model. Exception: {str(e)}")
            f.close()
            raise e

    def SaveModelArchitecture(self):

        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        if not os.path.exist('ModelArchitecture'):
            os.makedirs('ModelArchitecture')

        try:
            model = self.CreateModel()

            keras.utils.save_model(model, os.path.join('ModelArchitecture', 'ModelStructure.png'))
            Logger().log(f, 'Model structure saved as a png file successfully!')
            f.close()

        except Exception as e:
            Logger().log(f, f"Exception occured while saving the model structure as a png file. {str(e)}")
            raise e
    
    
