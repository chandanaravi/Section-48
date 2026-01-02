import os
from pathlib import Path
from dotenv import load_dotenv
from src.components.DataIngestion import DataIngestion
from src.applogger import logger
from src.components.DataTransformation import DataTransformation
from src.applogger import logger


from src.components.CSV_DataIngestion import CSV_DataIngestion
from src.components.DataTransformation import DataTransformation

from src.components.ModelTrainer import ModelTrainer_LinearRegression

load_dotenv()


class App:
    
    def Run():

        logger.info("==================== Application started ====================")
        # Load environment variables from .env file
        load_dotenv() 
        data_source_file_path = os.getcwd() + "/DataSets/stud.csv"
        #train_data_file_path, test_data_file_path, raw_data_file_path = DataIngestion(data_source_type="csv", source_data_file_path=data_source_file_path)
        dataingestion = DataIngestion(data_source_type="csv", source_data_file_path=data_source_file_path)
        train_data_file_path, test_data_file_path, raw_data_file_path=dataingestion.LoadDataFromCSV()

        data_trans_obj=DataTransformation(dataingestion)

        train_data,test_data,preprocessed_obj_file_path=data_trans_obj.initiate_data_transformation()

        logger.info("==================== Application finished ====================")
        
    def Run2(self):

        logger.info("==================== Application started ====================")
        file_path=os.getcwd()+"/DataSets/stud.csv"
        di=CSV_DataIngestion(file_path=file_path)
        di.Load()

        dt=DataTransformation(data_ingestion=di,target_column_name="math_score")
        final_train_data,final_test_data,preprocesser_pkl_lile=dt.initiate_data_transformation()
        X_train, y_train = final_train_data[:,:-1], final_train_data[:,-1]

        model=ModelTrainer_LinearRegression()
        model.TrainModel_and_Save(X_train=X_train,y_train=y_train)
        logger.info("==================== Application finished ====================")
        

if __name__ == "__main__":
    app=App()
    app.Run2()
     
    
   