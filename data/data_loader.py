from Logging.LoggerFile import Logger
from tensorflow import keras
import os



class DataLoader:


    def __init__(self):
        pass

    def getData(self):

        try:
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

            return X_train, y_train , X_test, y_test
        
        except Exception as e:
            raise e

    def getValidationData(self):

        if not os.path.exist('LoggedData'):
                os.makedirs('LoggedData')

        f = open(os.path.join('LoggedData', 'dataLoading.txt'), 'a+')

        try:
            Logger().log(f, "Loading data...")
            X_train_full, y_train_full , X_test, y_test = self.getData()

            X_valid, X_train = X_train_full[:10000]/255, X_train_full[10000:]/255
            y_valid, y_train = y_train_full[:10000], y_train_full[10000:]
            X_test = X_test[:]/255

            Logger().log(f, 'Data loading completed successfully!')

            f.close()

            return X_train, X_valid, X_test, y_train, y_valid, y_test

        except Exception as e:
            Logger().log(f, f"Exception occured while loading the data. Exception: {str(e)}")
            f.close()
            raise e

    
