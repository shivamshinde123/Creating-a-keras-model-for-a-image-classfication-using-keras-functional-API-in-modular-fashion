from datetime import datetime


class Logger:


    def __init__(self):
        pass

    def log(self, fileObject, message):


        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.time = self.now.strftime('%H:&M:%S')


            fileObject.write(str(self.date) + " " + str(self.time) + " ----> " + str(message))

        except Exception as e:
            raise e
