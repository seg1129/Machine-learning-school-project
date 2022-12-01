# Libraries we need for this project
import numpy as np
import pandas as pd
import os
import earthpy as et
import math
import statistics
from statistics import stdev
import requests
import os.path 
import csv 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

# data prep functions
class Data_prep2:

    def get_arrest_data(self):
        # I just picked the biggest dataset - this will take a few minutes to download. luckly you will only need to 
        # download this once
        file_url = "https://stacks.stanford.edu/file/druid:yg821jf8611/yg821jf8611_tn_nashville_2020_04_01.csv.zip"
        data_file = et.data.get_data(url=file_url)
        fname = os.path.join(data_file, "tn_nashville_2020_04_01.csv")
        return pd.read_csv(fname, on_bad_lines='skip')

    def shuffle_data(self, data, seed):
        np.random.seed(seed)
        try:
            np.random.shuffle(data.values)
        except:
            np.random.shuffle(data)
        return(data)

    # splits data into training(2/3) and validation(1/3) dataframes
    def split_data(self, data_frame):
        training_length = round(2/3 * len(data_frame))

        training_df = data_frame[:training_length]
        validation_df = data_frame[training_length:]

        return training_df, validation_df

    # assuming that that yhat column is the last column in the dataset, this returns the features seperated from
    # the yhat column. for example to use this function: 'arrest_feat_df, arrest_yhat_df = split_data_yhat(df)'
    def split_data_yhat(self, df):
        # This function will not work on our dataset - our yhat is 'arrest_made' column and it is not the 
        # last column of the dataset. so if someone needs this - you will have to re-write this function.
        return df.iloc[:,:-1], df.iloc[:,-1]
    
    def moveYcolumnToEnd(self, data, indexOfY):
        #takes a numpy array and moves column at "indexOfY" to the last column of the array.  
        #All columns at indexes greater than "indexOfY" are shifted to the left by one.
        columns = data.columns
        np_data = data.to_numpy()
        numOfRow, NumOfColumns = np_data.shape
        if(indexOfY < NumOfColumns):
            np_data[:, indexOfY:] = np.roll(np_data[:, indexOfY:], -1, 1)
        else:
            print("Your Y index is out of range, cannot move column")
        print("column type: ", type(columns))
        df_transf_data = pd.DataFrame(np_data, columns = columns)
        #df_transf_data.rename(columns = {'is_white', 'is_black', 'warning_issued'}'is_hispanic', 'is_asian'
        
        return df_transf_data
                              
    def move_y_col_end_df(self, data, column):
        column_to_move = data.pop(column)
        data[column] = column_to_move
        return data

    def get_arrest_data2(self):
        #gets the data from a local file if it exists
        if os.path.exists('tn_nashville_2020_04_01.csv'):
            print("File already exists, getting data locally.....")
            data = pd.read_csv('tn_nashville_2020_04_01.csv',on_bad_lines='skip', dtype=str)
        else:
            file_url = 'https://stacks.stanford.edu/file/druid:yg821jf8611/yg821jf8611_tn_nashville_2020_04_01.csv.zip'
            data_file = et.data.get_data(url=file_url)
            fname = os.path.join(data_file, 'tn_nashville_2020_04_01.csv')
            data = pd.read_csv(fname, on_bad_lines='skip', dtype=str)
        columns = list(data)
        print("columns: ", columns)
        print("data shape after import: ", data.shape)
        return data

    def encodeFeatures(self, dataframe,skipColumnsList=[]):
        le = preprocessing.LabelEncoder()
        for column in dataframe.columns:
            if column not in skipColumnsList:
                # Converting string labels into numbers.
                dataframe[column]=le.fit_transform(dataframe[column])

        return dataframe

    def replaceNulls(self, data):
    #takes in a numpy array and replaces missing null values with the most frequent values in that column
        columns = data.columns
        print("columns: ", columns)
        print("data shape: ", data.shape)
        np_arrest_data = data.to_numpy()
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(np_arrest_data)
        SimpleImputer()
        transf_data = imp.transform(np_arrest_data)
        df_transf_data = pd.DataFrame(transf_data, columns = columns)
        
        return df_transf_data

    def truncateTime(self, time):
        #truncates time values to just the hour   
        return str(time)[0:2]

    def convertAgeFeature(self, stringAge):
        young =0
        middle = 1
        old =2
        convertedAge = 0
        age = float(stringAge)
        if  age >= 25:
            if age < 50:
                convertedAge = 1
            else:
                convertedAge = 2
       
        return convertedAge

    def seperate_race_columns(self, arrest_data):
        arrest_data['is_white'] = arrest_data['subject_race'].apply(lambda x : x == 'white')
        
        
        arrest_data['is_black'] = arrest_data['subject_race'].apply(lambda x : x == 'black')

        
        arrest_data['is_hispanic'] = arrest_data['subject_race'].apply(lambda x : x == 'hispanic')

        
        arrest_data['is_asian'] = arrest_data['subject_race'].apply(lambda x : x == 'asian/pacific islander')

        
        arrest_data.set_index('is_white').columns
        arrest_data.set_index('is_black').columns
        arrest_data.set_index('is_hispanic').columns
        arrest_data.set_index('is_asian').columns
        
        del arrest_data['subject_race']
        return arrest_data

    def convert_race_boolean(self, race):
        if race == 'white':
            return 0
        else:
            return 1
        
        

    def getPreprocessedArrestData(self):
        arrest_dataFrame = self.get_arrest_data2().copy()
        # arrest_dataFrame = arrest_dataFrame.dropna()
        # arrest_data = arrest_dataFrame[['subject_race', 'subject_sex','subject_age','time','violation','frisk_performed','search_vehicle','arrest_made']].copy()
        arrest_data = arrest_dataFrame[['subject_race', 'subject_sex','subject_age','time','violation','frisk_performed','search_vehicle','warning_issued']].copy()
        arrest_data = self.replaceNulls(arrest_data)
        arrest_data = self.seperate_race_columns(arrest_data)
        
        # arrest_data = arrest_dataFrame[['subject_sex', 'subject_age', 'time', 'violation', 'frisk_performed', 'search_vehicle', 'is_white', 'is_black', 'is_hispanic', 'is_asian', 'warning_issued']].copy()
        # yhat_index_no = arrest_data.columns.get_loc('warning_issued')
        # print("yhat index number: ", yhat_index_no)
        # arrest_data = self.moveYcolumnToEnd(arrest_data, yhat_index_no)
        arrest_data = self.move_y_col_end_df(arrest_data, 'warning_issued')
        print("final column order: ", arrest_data.columns)
        #print(arrest_data.head(10))
        #arrest_data = truncateTime(data)
        arrest_data['time'] = arrest_data['time'].apply(lambda x : self.truncateTime(x))
        arrest_data['subject_age'] = arrest_data['subject_age'].apply(lambda x : self.convertAgeFeature(x))
        # arrest_data['subject_race'] = arrest_data['subject_race'].apply(lambda x : self.convert_race_boolean(x))
        #skipColumnsList=['time','subject_age']
        arrest_data = self.encodeFeatures(arrest_data)   
        #data = data.to_numpy()
        #arrest_data=replaceNulls(data)
        return arrest_data

    def getPreprocessedArrestDataWithoutRace(self):
        arrest_dataFrame = self.get_arrest_data2().copy()
        # arrest_dataFrame = arrest_dataFrame.dropna()
        arrest_data = arrest_dataFrame[['subject_sex','subject_age','time','violation','frisk_performed','search_vehicle','arrest_made']].copy()
        arrest_data = arrest_dataFrame[['subject_race', 'subject_sex','subject_age','time','violation','frisk_performed','search_vehicle','warning_issued']].copy()
        # arrest_data = self.replaceNulls(arrest_data)
        #arrest_data = truncateTime(data)
        arrest_data['time'] = arrest_data['time'].apply(lambda x : self.truncateTime(x))
        arrest_data['subject_age'] = arrest_data['subject_age'].apply(lambda x : self.convertAgeFeature(x))
        #skipColumnsList=['time','subject_age']
        arrest_data = self.encodeFeatures(arrest_data)    
        #data = data.to_numpy()
        #arrest_data=replaceNulls(data)
        return arrest_data

        
