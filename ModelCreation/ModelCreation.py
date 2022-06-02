from tensorflow import keras
from data.data_loader import DataLoader
from Logging.LoggerFile import Logger
import os
import pandas as pd
import numpy as np

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
            model = keras.Model(inputs =[self.input_layer], outputs=[self.output_layer])
            Logger.log(f, 'Model created successfully!')
            f.close()
            return model

        except Exception as e:
            Logger().log(f, f"Exception occured while creating layers or creating model. Exception: {str(e)}")
            f.close()
            raise e

    def SaveModelArchitecture(self, model):

        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        if not os.path.exist('ModelArchitecture'):
            os.makedirs('ModelArchitecture')

        try:
            model = model

            keras.utils.save_model(model, os.path.join('ModelArchitecture', 'ModelStructure.png'))
            Logger().log(f, 'Model structure saved as a png file successfully!')
            f.close()

        except Exception as e:
            Logger().log(f, f"Exception occured while saving the model structure as a png file. {str(e)}")
            raise e

    def CompilingModel(self, model):

        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        try:
            
            model = model
            Logger().log(f, 'Compiling the model...')
            model.compile(loss='sparse_categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
            Logger.log(f, 'Model compiled successfully!')
            f.close()
            return model

        except Exception as e:
            Logger().log(f, f"Exception occured while compiling the model. Exception: {str(e)}")
            f.close()
            raise e
            


    def FittingModel(self, epochs=20, model=None):

        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        try:

            model = model
            Logger().log(f, 'Fitting the model...')
            history = model.fit(self.X_train, self.y_train,epochs=epochs, 
                                        validation_data = (self.X_valid, self.y_valid) )

            Logger().log(f, 'Model fitted successfully!')

            if not os.path.exists('Plots'):
                os.makedirs('Plots')
            
            history_df = pd.DataFrame(history.history)
            figure = history_df.plot(figsize=(12,6)).get_figure()

            figure.savefig(os.path.join('Plots', 'LossAndAccuracyVSepochsPlot'))
            Logger.log(f, 'Saved the plot between the loss and accuracy of train and validation set with respect to number of epochs')
            f.close()
            return model

        except Exception as e:
            Logger().log(f, f"Exception occured while fitting the model to the training data. Exception: {str(e)}")
            f.close()
            raise e

    
    def ModelEvaluationOnTestData(self, model):
        
        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'ModelCreation.txt'), 'a+')

        try:
            self.model = model

            model_loss, model_accuracy = self.model.evaluate(self.X_test, self.y_test)[0], self.model.evaluate(self.X_test, self.y_test)[1]

            Logger().log(f, f"Model's loss on the test data: {np.round(model_loss,3)}")
            Logger().log(f, f"Model's accuracy on the test data: {np.round(model_accuracy,3)}")

            f.close()

            print(f"Model's loss on the test data: {np.round(model_loss,3)}")
            print(f"Model's accuracy on the test data: {np.round(model_accuracy,3)}")

        except Exception as e:
            Logger().log(f, f"Exception occured while evaluating the fitted model on the test data. Exception: {str(e)}")
            f.close()
            raise e

    
    def PredictionForNewData(self, input_array):

        if not os.path.exists('LoggedData'):
            os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'PredictionForNewData.txt'), 'a+')

        try:
            y_proba = self.model.predict(input_array)
            y_pred = np.argmax(y_proba, axis=1)

            Logger().log(f, f"Predicted value for the input array {input_array} is {y_pred}")

            print(f"Predicted value for the input array {input_array} is {y_pred}")

            f.close()

        except Exception as e:
            Logger().log(f, f"Exception occured while predicting the output for the new picture. Exception: {str(e)}")
            f.close()
            f.close()

    