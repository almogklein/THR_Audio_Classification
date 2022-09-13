import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from EDA import EDA
from DS import DS
from Extract_Features import Extract_Features
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from catboost import CatBoostClassifier

if __name__ == '__main__':
    '''
    chose can be:
    EDA - to start EDA process
    Extract_Features - Feature extraction (FFT)
    Extract_hits_Features - signal feature extraction (Mfcc, Zero-Cross-Rate, etc..)
    DS - to start training a classification model
    '''
    chose = 'DS' # 'EDA'  'DS'

    features = pd.read_csv("features_data_.csv")
    features_fft = pd.read_csv("features_data_FFT.csv")
    data = pd.read_csv("DATA.csv")

    garbeg = ['broach_495 hit_index_0', 'broach_495 hit_index_1', 'broach_495 hit_index_2', 'broach_495 hit_index_3',
              'broach_450 hit_index_5 ', 'broach_450 hit_index_6', 'broach_450 hit_index_7', 'broach_450 hit_index_8',
              'broach_450 hit_index_12' , 'broach_450 hit_index_13' , 'broach_450 hit_index_14', 'broach_375 hit_index_0',
              'broach_375 hit_index_3', 'broach_375 hit_index_6', 'broach_375 hit_index_10', 'broach_375 hit_index_11',
              'broach_375 hit_index_12', 'broach_375 hit_index_13', 'broach_375 hit_index_14', 'broach_375 hit_index_15',
              'broach_375 hit_index_16', 'broach_375 hit_index_17' , 'broach_375 hit_index_18', 'broach_375 hit_index_19', 'broach_375 hit_index_20',
              'broach_375 hit_index_21', 'broach_375 hit_index_22', 'broach_375 hit_index_23', 'broach_375 hit_index_24', 'broach_375 hit_index_25',
              'broach_375 hit_index_34', 'broach_385 hit_index_0' , 'broach_385 hit_index_1', 'broach_385 hit_index_2', 'broach_385 hit_index_3',
              'broach_385 hit_index_4', 'broach_385 hit_index_5', 'broach_385 hit_index_6' , 'broach_385 hit_index_7', 'broach_385 hit_index_8',
              'broach_385 hit_index_9', 'broach_385 hit_index_10', 'broach_385 hit_index_11', 'broach_385 hit_index_12', 'broach_385 hit_index_13',
              'broach_385 hit_index_14', 'broach_385 hit_index_15', 'broach_385 hit_index_16', 'broach_385 hit_index_17', 'broach_385 hit_index_21',
              'broach_385 hit_index_22', 'broach_385 hit_index_23', 'broach_385 hit_index_24', 'broach_385 hit_index_25','broach_385 hit_index_26',
              'broach_398 hit_index_0', 'broach_398 hit_index_1' , 'broach_398 hit_index_2', 'broach_398 hit_index_3' , 'broach_398 hit_index_7', 'broach_398 hit_index_8',
              'broach_485 hit_index_25', 'broach_392 hit_index_3', 'broach_392 hit_index_4', 'broach_392 hit_index_5', 'broach_392 hit_index_6',
              'broach_392 hit_index_8', 'broach_392 hit_index_9', 'broach_392 hit_index_10', 'broach_392 hit_index_11', 'broach_392 hit_index_12',
              'broach_392 hit_index_13' , 'broach_392 hit_index_14', 'broach_392 hit_index_15' ,'broach_392 hit_index_16', 'broach_392 hit_index_17',
              'broach_392 hit_index_18', 'broach_392 hit_index_19', 'broach_392 hit_index_20', 'broach_392 hit_index_21', 'broach_392 hit_index_22',
              'broach_392 hit_index_23', 'broach_392 hit_index_24', 'broach_392 hit_index_25', 'broach_392 hit_index_26', 'broach_392 hit_index_27',
              'broach_392 hit_index_28', 'broach_392 hit_index_29', 'broach_392 hit_index_30', 'broach_392 hit_index_31', 'broach_392 hit_index_32',
              'broach_392 hit_index_33', 'broach_392 hit_index_34', 'broach_392 hit_index_35', 'broach_392 hit_index_36',
              'broach_392 hit_index_37', 'broach_392 hit_index_38', 'broach_392 hit_index_39', 'broach_392 hit_index_40', 'broach_392 hit_index_45',
              'broach_392 hit_index_46', 'broach_392 hit_index_48', 'broach_392 hit_index_49', 'broach_392 hit_index_50',
              'broach_392 hit_index_51', 'broach_392 hit_index_52', 'broach_392 hit_index_53', 'broach_392 hit_index_54', 'broach_392 hit_index_60',
              'broach_392 hit_index_61', 'broach_614 hit_index_13', 'broach_614 hit_index_12', 'broach_614 hit_index_11', 'broach_614 hit_index_10',
              'broach_614 hit_index_9', 'broach_614 hit_index_8', 'broach_614 hit_index_7', 'broach_608 hit_index_2', 'broach_585 hit_index_2',
              'broach_585 hit_index_37', 'broach_585 hit_index_38', 'broach_585 hit_index_41', 'broach_585 hit_index_42', 'broach_585 hit_index_43',
              'broach_576 hit_index_1', 'broach_576 hit_index_2', 'broach_576 hit_index_3', 'broach_548 hit_index_0', 'broach_548 hit_index_1',
              'broach_548 hit_index_2', 'broach_548 hit_index_3', 'broach_548 hit_index_8', 'broach_471 hit_index_0', 'broach_471 hit_index_3',
              'broach_471 hit_index_4', 'broach_471 hit_index_5', 'broach_471 hit_index_6', 'broach_471 hit_index_7', 'broach_471 hit_index_8', 'broach_471 hit_index_9',
              'broach_471 hit_index_10', 'broach_471 hit_index_14', 'broach_471 hit_index_15', 'broach_471 hit_index_16', 'broach_471 hit_index_17',
              'broach_471 hit_index_18', 'broach_471 hit_index_19', 'broach_471 hit_index_20', 'broach_471 hit_index_21', 'broach_517 hit_index_2',
              'broach_517 hit_index_3' , 'broach_517 hit_index_6', 'broach_517 hit_index_7' , 'broach_517 hit_index_18', 'broach_517 hit_index_19',
              'broach_517 hit_index_20', 'broach_517 hit_index_21' 'broach_467 hit_index_1', 'broach_467 hit_index_2', 'broach_467 hit_index_3' , 'broach_467 hit_index_7',
              'broach_467 hit_index_17', 'broach_508 hit_index_14', 'broach_508 hit_index_15', 'broach_508 hit_index_16', 'broach_508 hit_index_17',
              'broach_373 hit_index_18', 'broach_387 hit_index_3', 'broach_412 hit_index_0', 'broach_412 hit_index_6', 'broach_412 hit_index_7',
              'broach_412 hit_index_8', 'broach_412 hit_index_16', 'broach_422 hit_index_22', 'broach_422 hit_index_23', 'broach_422 hit_index_24',
              'broach_422 hit_index_29', 'broach_422 hit_index_31', 'broach_422 hit_index_33', 'broach_483 hit_index_3', 'broach_483 hit_index_4',
              'broach_483 hit_index_5', 'broach_485 hit_index_0', 'broach_485 hit_index_1' , 'broach_485 hit_index_2' , 'broach_485 hit_index_3'
              , 'broach_485 hit_index_4' , 'broach_485 hit_index_5' , 'broach_485 hit_index_6' , 'broach_485 hit_index_7' , 'broach_485 hit_index_8'
              , 'broach_485 hit_index_9' , 'broach_485 hit_index_10', 'broach_485 hit_index_11', 'broach_485 hit_index_12', 'broach_485 hit_index_13'
              , 'broach_485 hit_index_14', 'broach_485 hit_index_15', 'broach_485 hit_index_16', 'broach_485 hit_index_17', 'broach_485 hit_index_18'
              , 'broach_485 hit_index_19', 'broach_485 hit_index_20', 'broach_485 hit_index_21', 'broach_485 hit_index_22', 'broach_485 hit_index_23'
              , 'broach_485 hit_index_24', 'broach_485 hit_index_25', 'broach_485 hit_index_26', 'broach_485 hit_index_27', 'broach_485 hit_index_28'
              , 'broach_485 hit_index_29', 'broach_485 hit_index_30', 'broach_485 hit_index_31', 'broach_485 hit_index_32', 'broach_485 hit_index_33'
              , 'broach_485 hit_index_34', 'broach_485 hit_index_35', 'broach_485 hit_index_36', 'broach_485 hit_index_37', 'broach_485 hit_index_38'
              , 'broach_485 hit_index_39', 'broach_485 hit_index_40', 'broach_500 hit_index_7', 'broach_506 hit_index_7', 'broach_506 hit_index_31',
              'broach_506 hit_index_32', 'broach_506 hit_index_33', 'broach_506 hit_index_39', 'broach_506 hit_index_40', 'broach_506 hit_index_41',
              'broach_506 hit_index_42', 'broach_524 hit_index_3', 'broach_524 hit_index_4', 'broach_526 hit_index_1', 'broach_526 hit_index_2',
              'broach_558 hit_index_0', 'broach_558 hit_index_2', 'broach_558 hit_index_5', 'broach_558 hit_index_6','broach_558 hit_index_7',
              'broach_558 hit_index_8', 'broach_558 hit_index_9', 'broach_558 hit_index_10', 'broach_558 hit_index_11', 'broach_558 hit_index_15',
              'broach_558 hit_index_16', 'broach_574 hit_index_5', 'broach_567 hit_index_0',  'broach_567 hit_index_1',  'broach_567 hit_index_2',
              'broach_567 hit_index_3', 'broach_567 hit_index_4', 'broach_567 hit_index_5', 'broach_567 hit_index_6',  'broach_567 hit_index_7',
              'broach_567 hit_index_8', 'broach_567 hit_index_9', 'broach_567 hit_index_10', 'broach_567 hit_index_11', 'broach_567 hit_index_12',
              'broach_567 hit_index_13','broach_585 hit_index_37', 'broach_585 hit_index_38', 'broach_585 hit_index_39', 'broach_585 hit_index_41',
              'broach_585 hit_index_42' ,'broach_599 hit_index_0', 'broach_599 hit_index_1', 'broach_599 hit_index_2', 'broach_599 hit_index_3', 'broach_599 hit_index_4',
              'broach_599 hit_index_7', 'broach_599 hit_index_8', 'broach_599 hit_index_9', 'broach_599 hit_index_10', 'broach_599 hit_index_11' ,'broach_599 hit_index_12',
              'broach_599 hit_index_13', 'broach_599 hit_index_18', 'broach_599 hit_index_21', 'broach_599 hit_index_22','broach_599 hit_index_23','broach_599 hit_index_24',
              'broach_599 hit_index_25','broach_599 hit_index_26', 'broach_599 hit_index_27', 'broach_599 hit_index_28','broach_601 hit_index_0','broach_601 hit_index_1',
              'broach_601 hit_index_2','broach_601 hit_index_3','broach_601 hit_index_5', 'broach_601 hit_index_6','broach_601 hit_index_7','broach_601 hit_index_8','broach_601 hit_index_9',
              'broach_605 hit_index_7','broach_605 hit_index_8','broach_605 hit_index_29', 'broach_605 hit_index_30','broach_605 hit_index_39','broach_605 hit_index_40','broach_605 hit_index_41',
              ]

    features = features[~features['Unnamed: 0'].isin(garbeg)]
    features_fft = features_fft[~features_fft['Unnamed: 0'].isin(garbeg)]
    data = data[~data['Unnamed: 0'].isin(garbeg)]

    if chose == 'Extract_Features' or chose == 'Extract_hits_Features':

        sample_rate = 64000
        HOP_LENGTH_DIVISION_NUMBER = 7
        N_FFT_DIVISION_NUMBER = 3.5
        EF1 = Extract_Features()

        if chose == 'Extract_Features':
            path = r"C:\Users\python3\Downloads\raw data"
            CSV_READ = True
            PLOT = False
            EF1.load_metadata_database(path, CSV_READ, PLOT, sample_rate,
                                       HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)

        if chose == 'Extract_hits_Features':
            path = r"C:\Users\python3\Downloads\all_hits"
            EF1.check_hits(path, sample_rate, HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)

    if chose == 'DS':

        EDA_ = EDA(features_fft, features, data) ##  Data slicing using an algorithm - time-frequency characteristics
        final_data = EDA_.mergeDatasets() #Merging the
        DS_ = DS(final_data)
        final_data = DS_.parseData(1.0, 0.5) # early p - first p% , late p - final p%


        print('-------------------all final data loaded--------------------')

        final_data = DS_.stat_tests(final_data, a=0.05, manual=False)

        # final_data.to_csv('final_data_gg_1.0_0.5.csv')

        print('-------------------statr modeling Data slicing - algorithmic -------------------')

        '''
            GradientBoostingClassifier(loss='exponential', learning_rate=0.01, min_samples_split=4, max_features='log2',
                                       max_depth=20, n_estimators=50),
            GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=6, max_features='log2',
                                       max_depth=20, n_estimators=100),
            GradientBoostingClassifier(loss='exponential', learning_rate=0.4, min_samples_split=3, max_depth=20,
                                       max_features='log2', n_estimators=150),

            XGBClassifier(verbosity=2, eta=0.5, gamma=1.5, max_depth=20, sampling_method='uniform', n_estimators=50,
                          reg_lambda=2.0, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.5, gamma=5.3, max_depth=30, sampling_method='gradient_based',
                          mean_child_weight=2, n_estimators=70, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.5, gamma=5.3, max_depth=30, sampling_method='gradient_based',
                          n_estimators=70, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.3, gamma=1.8, max_depth=60, sampling_method='gradient_based',
                          n_estimators=70, class_weight='balanced'),


            LogisticRegression(penalty='l1', C=0.2, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.3, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.5, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.2, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.3, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.5, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.2, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.3, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.5, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.3, solver='lbfgs', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', class_weight='balanced'),

            SVC(C=0.5, probability=True, kernel='poly', degree=5, class_weight='balanced'),
            SVC(C=3.0, probability=True, kernel='poly', degree=8, class_weight='balanced'),
            SVC(C=5.0, probability=True, kernel='poly', degree=10, class_weight='balanced'),
            SVC(C=0.5, probability=True, kernel='rbf', degree=5, class_weight='balanced'),
            SVC(C=3.0, probability=True, kernel='rbf', degree=8, class_weight='balanced'),
            SVC(C=5.0, probability=True, kernel='rbf', degree=10, class_weight='balanced'),
            SVC(C=0.5, probability=True, kernel='sigmoid', degree=5, class_weight='balanced'),
            SVC(C=3.0, probability=True, kernel='sigmoid', degree=8, class_weight='balanced'),
            SVC(C=5.0, probability=True, kernel='sigmoid', degree=10, class_weight='balanced'),

            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=4, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=6, class_weight='balanced'),
            RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=3,class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=40, min_samples_split=3, class_weight='balanced'),

        
            ===============================================
            GradientBoostingClassifier(loss='exponential', learning_rate=0.01, min_samples_split=4, max_features='log2',
                                       max_depth=20, n_estimators=50),
            GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=6, max_features='log2',
                                       max_depth=20, n_estimators=100),
            GradientBoostingClassifier(loss='exponential', learning_rate=0.4, min_samples_split=3, max_depth=20,
                                       max_features='log2', n_estimators=150),

            XGBClassifier(verbosity=2, eta=0.5, gamma=1.5, max_depth=20, sampling_method='uniform', n_estimators=50,
                          reg_lambda=2.0, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.5, gamma=5.3, max_depth=30, sampling_method='gradient_based',
                          mean_child_weight=2, n_estimators=70, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.5, gamma=5.3, max_depth=30, sampling_method='gradient_based',
                          n_estimators=70, class_weight='balanced'),
            XGBClassifier(verbosity=2, eta=0.3, gamma=1.8, max_depth=60, sampling_method='gradient_based',
                          n_estimators=70, class_weight='balanced'),

            LogisticRegression(penalty='l1', C=0.2, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.3, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.5, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.2, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.3, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.5, solver='liblinear', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.2, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.3, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l1', C=0.5, solver='saga', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.3, solver='lbfgs', class_weight='balanced'),
            LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', class_weight='balanced'),

            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3,
                                   class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=4, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=6, class_weight='balanced'),
            RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=3, class_weight='balanced'),
            RandomForestClassifier(n_estimators=150, max_depth=40, min_samples_split=3, class_weight='balanced'),
        
        
    
        '''

        models = [
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.01, min_samples_split=4, max_features='log2',
            #                            max_depth=20, n_estimators=50),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=6, max_features='log2',
            #                            max_depth=20, n_estimators=100),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.4, min_samples_split=3, max_depth=20,
            #                            max_features='log2', n_estimators=150),
            #
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.5, max_depth=5, sampling_method='uniform', n_estimators=50,
            #               reg_lambda=50, class_weight='balanced', subsample=0.7),
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.3, max_depth=5, sampling_method='uniform',
            #               mean_child_weight=2, reg_lambda=100, n_estimators=50, class_weight='balanced', subsample=0.7),
            # XGBClassifier(verbosity=1, eta=0.5, reg_lambda=150, gamma=0.3, max_depth=5, sampling_method='uniform',
            #               n_estimators=50, class_weight='balanced', subsample=0.7),
            # XGBClassifier(verbosity=1, eta=0.3, reg_lambda=200, gamma=1, max_depth=5, sampling_method='uniform',
            #               n_estimators=50, class_weight='balanced', subsample=0.7),
            #
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.5, max_depth=5, sampling_method='uniform', n_estimators=100,
            #                 class_weight='balanced', subsample=0.7, reg_alpha=75),
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.3, max_depth=7, sampling_method='uniform',
            #               mean_child_weight=3, n_estimators=50, class_weight='balanced', subsample=0.7,
            #               reg_alpha=100),
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.3, max_depth=8, sampling_method='uniform',
            #               mean_child_weight=3, n_estimators=50, class_weight='balanced', subsample=0.6,
            #               reg_alpha=65),

            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.3, max_depth=9, sampling_method='uniform',
            #               mean_child_weight=3, n_estimators=50, class_weight='balanced', subsample=0.5,
            #               reg_alpha=50),
            #
            # XGBClassifier(verbosity=1, eta=0.5, gamma=0.3, max_depth=10, sampling_method='uniform',
            #               mean_child_weight=3, n_estimators=50, class_weight='balanced', subsample=0.6,
            #               reg_alpha=50),
            # XGBClassifier(verbosity=1, eta=0.7, gamma=3.0, max_depth=11, sampling_method='uniform',
            #               mean_child_weight=3, n_estimators=50, class_weight='balanced', subsample=0.7,
            #               reg_alpha=50),

           #STARTTT


            # XGBClassifier(verbosity=1, eta=0.5, gamma=1.0, max_depth=15, sampling_method='uniform',
            #               mean_child_weight=6, n_estimators=100, class_weight='balanced', subsample=0.8,
            #               reg_alpha=53),
            # XGBClassifier(verbosity=1, eta=0.5, gamma=1.0, max_depth=20, sampling_method='uniform',
            #               mean_child_weight=6, n_estimators=100, class_weight='balanced', subsample=0.8,
            #               reg_alpha=55),
            # XGBClassifier(verbosity=1, eta=0.5, gamma=3.0, max_depth=25, sampling_method='uniform',
            #               mean_child_weight=6, n_estimators=100, class_weight='balanced', subsample=0.7,
            #               reg_alpha=57),

            # LogisticRegression(penalty='l1', C=80, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=100, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=100, solver='lbfgs', class_weight='balanced'),

            XGBClassifier(verbosity=1, eta=0.5, gamma=1.0, max_depth=12, sampling_method='uniform',
                          mean_child_weight=6, n_estimators=50, class_weight='balanced', subsample=0.7,
                          reg_alpha=50),

            LogisticRegression(penalty='elasticnet', C=100, solver='saga', class_weight='balanced', l1_ratio=0.5),

            RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_split=3,
                                   class_weight='balanced',
                                   min_impurity_decrease=0.05),

            GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=3, max_features='log2',
                                       max_depth=4, n_estimators=5, min_impurity_decrease=0.05, subsample=0.2)
            # RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_split=2,
            #                        class_weight='balanced',
            #                        min_impurity_decrease=0.1),
            #
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=5, max_features='log2',
            #                            max_depth=4, n_estimators=5, min_impurity_decrease=0.2, subsample=0.2),
            #
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=4, max_features='log2',
            #                            max_depth=4, n_estimators=5, min_impurity_decrease=0.2, subsample=0.2),
            #
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.2, min_samples_split=3, max_features='log2',
            #                            max_depth=4, n_estimators=5, min_impurity_decrease=0.5, subsample=0.2),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.3, min_samples_split=4, max_features='log2',
            #                            max_depth=5, n_estimators=5, min_impurity_decrease=0.2, subsample=0.2),


            # XGBClassifier(verbosity=1, eta=0.5,  gamma=0.3, max_depth=5, sampling_method='uniform',
            #               n_estimators=50, class_weight='balanced', subsample=0.7, reg_alpha=150),
            # XGBClassifier(verbosity=1, eta=0.3,  gamma=1, max_depth=5, sampling_method='uniform',
            #               n_estimators=50, class_weight='balanced', subsample=0.7, reg_alpha=200)
            #
            # LogisticRegression(penalty='l1', C=0.2, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=0.3, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=0.5, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.2, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.3, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.5, solver='liblinear', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=0.2, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=0.3, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=0.5, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.3, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', class_weight='balanced'),
            #
            # RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
            #                        class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            # RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3,
            #                        class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            # RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=3,
            #                        class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            # RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3,
            #                        class_weight='balanced_subsample', bootstrap=True, oob_score=True),
            # RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=3, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=4, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=6, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=3, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=7, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=3, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=3, class_weight='balanced'),
            # RandomForestClassifier(n_estimators=150, max_depth=40, min_samples_split=3, class_weight='balanced'),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.1, min_samples_split=4, max_features='log2',
            #                            max_depth=10, n_estimators=5, min_impurity_decrease=0.1, subsample=0.1),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.01, min_samples_split=10, max_features='log2',
            #                            max_depth=20, n_estimators=10, min_impurity_decrease=0.1, subsample=0.1),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.01, min_samples_split=10, max_features='log2',
            #                            max_depth=10, n_estimators=5, min_impurity_decrease=0.1, subsample=0.1),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.02, min_samples_split=6, max_features='log2',
            #                            max_depth=5, n_estimators=30, min_impurity_decrease=0.1, subsample=0.1),
            # GradientBoostingClassifier(loss='exponential', learning_rate=0.02, min_samples_split=6, max_features='log2',
            #                            max_depth=3, n_estimators=20, min_impurity_decrease=0.1, subsample=0.1),

            #
            # XGBClassifier(verbosity=1, eta=0.5, gamma=5.3, max_depth=30, sampling_method='gradient_based',
            #               n_estimators=70, class_weight='balanced'),
            #
            # RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced')
            # LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=5, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=10, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=50, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=1, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=5, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=10, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l1', C=50, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='elasticnet', C=1, solver='saga', class_weight='balanced', l1_ratio=0.5),
            # LogisticRegression(penalty='elasticnet', C=5, solver='saga', class_weight='balanced', l1_ratio=0.5),
            # LogisticRegression(penalty='elasticnet', C=10, solver='saga', class_weight='balanced', l1_ratio=0.5),
            # LogisticRegression(penalty='elasticnet', C=50, solver='saga', class_weight='balanced', l1_ratio=0.5),
            # LogisticRegression(penalty='l1', C=100, solver='saga', class_weight='balanced'),
            # LogisticRegression(penalty='l2', C=100, solver='lbfgs', class_weight='balanced'),
            # LogisticRegression(penalty='elasticnet', C=100, solver='saga', class_weight='balanced', l1_ratio=0.5)

        ]

        # final_data = pd.read_csv('final_data_gg_1.0_0.5.csv')
        # final_data.pop('Unnamed: 0')

        DS_ = DS(final_data)
        res_train, res_val, fitted_models, y_test, x_test = DS_.realStartModelinPleaseStandUp(final_data, models)
        # res_train, res_val, fitted_models = DS_.realStartModelinPleaseStandUp(final_data, models)

        print('-------------------start ploting - algorithmic--------------------')
        #DS.plotAcc(dict1_FFT, 1, 'Data slicing - algorithmic - FFT+features')
        DS_.plotAuc(res_train, res_val, fitted_models, y_test, x_test, final_data)
        # DS_.plotAuc(res_train, res_val, fitted_models, final_data)
