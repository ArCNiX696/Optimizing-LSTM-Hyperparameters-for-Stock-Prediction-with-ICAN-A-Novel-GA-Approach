import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from plot import *


class Preprocessing:
    def __init__(self,open_path,save_path,windows_size,save=False,normalize=False,standard=False,lstm=False):
        self.path=open_path
        self.save_path=save_path
        self.save=save
        self.lstm=lstm
        self.df=None
        self.scaler=StandardScaler()
        self.normalizer=MinMaxScaler()
        self.normalized=normalize
        self.standardized=standard
        self.windows_size = windows_size
        

    def open_csv(self,verbose=False):
        self.df= pd.read_csv(self.path)

        if verbose == True:
            print(f'\n{self.df.head(10)}\n') 
        
        return self.df
    
    
    def normalize(self,df,verbose=False):
        df_normalized=self.normalizer.fit_transform(df)

        df=pd.DataFrame(df_normalized, columns=df.columns ,index=df.index)

        if verbose==True:
            print(f'\nThis is the Normalized dataset:\n{df.head(50)}\n')

        return df
    
    def standardize(self,df,verbose=False):
        df_standardized=self.scaler.fit_transform(df)

        df=pd.DataFrame(df_standardized , columns=df.columns , index=df.index)

        if verbose==True:
            print(f'\nThis is the Standardized dataset:\n{df.head(50)}\n')

        return df
    
    def drop_column(self,column_name=str,verbose=True):
        # self.df=self.df.drop(column_name,axis=1)
        self.df.drop(column_name,axis=1 , inplace=True)
        
        if verbose == True:
            print(f"Column : {column_name} was removed from the dataset")
        
        return self.df
    
    def set_col_index(self,column_name=str,verbose=True):
        # self.df=self.df.set_index([column_name] , inplace=True)
        self.df.set_index([column_name] , inplace=True)

        if verbose == True:
            print(f"Column : {column_name} was set as the index of the dataset")

        # print(f'\n{self.df.head(10)}\n') 
        return self.df

    
    def analize_df(self,verbose=False,corr_plot=False,plot_corr_scatt=False,plot_box =False):
        #Make a comparison of with column with repect to the date , this should be done before any preprocessing step.
        if verbose == True:
            plot_datetime(self.df,'Date','Close')

        # self.df_numeric = self.df.select_dtypes(include=['float64','int64'])
        self.df_numeric = self.set_col_index('Date')
        
        if self.normalized == True:
            self.df_numeric = self.normalize(self.df_numeric,verbose=True)

        if self.standardized == False:
            self.df_numeric = self.standardize(self.df_numeric,verbose=True)

        
        corr_matrix=self.df_numeric.corr()

        if not self.df_numeric.empty and verbose ==True:
            self.df.info()
            print(f'\nNumber of Null values per column:\n{self.df.isnull().sum()}\n')
            print(f'\n{self.df.describe()}\n') #Describe all the columns even those that are not numerical
            print(f'\n{corr_matrix}\n')

        if corr_plot == True:
            plot_corr_matrix(corr_matrix)

        if plot_corr_scatt == True:
            plot_corr_scatter(self.df_numeric)

        if plot_box == True:
            boxplot(self.df_numeric,'Close')

        return self.df_numeric
    
    
    def lstm_preprocessing(self,df,windows_size,pre_y=str,verbose=False):#

        df = pd.DataFrame(df[pre_y].copy())

        for i in range(1,windows_size +1):
            df.loc[:,f'{pre_y}(t-{i})'] = df[pre_y].shift(i)

        df.dropna(inplace=True)


        if verbose==True:
            print(f'\nThis lstm dataset:\n{df.head(50)}\n')
            
        return df

        
    def run_preprocessing(self):
        self.open_csv(verbose=True)
        self.drop_column('Volume')
        df=self.analize_df(verbose=False,corr_plot=False,plot_corr_scatt=False,plot_box=False)

        if self.lstm == True:
            df=self.lstm_preprocessing(df,14,'Close',verbose=True)

        if self.save == True:
            df.to_csv(self.save_path, index=False)


if __name__ == '__main__':

    raw_folder='./Datasets'
    raw_data='TESLA.csv'
    open_path=os.path.join(raw_folder,raw_data)
    file_name,ext = os.path.splitext(raw_data)
    windows_size=7
    preprocessed_data= file_name +f' stock price dataset - windows size of  t-{windows_size} days,Normalized'+ext

    save_folder='./input datasets'
    save_path=os.path.join(save_folder,preprocessed_data)
    pre=Preprocessing(open_path,save_path,windows_size,save=True,normalize=True,standard=True,lstm=True)

    pre.run_preprocessing()





