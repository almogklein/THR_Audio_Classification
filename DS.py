import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy.stats import mannwhitneyu, norm
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams["figure.autolayout"] = True
import math
import seaborn as sns


class DS:
    def __init__(self, final_data):
        self.final_data = final_data

    def parseData(self, p_early, p_late):

        '''

        this function will extract and label (0 - early , 1 - late) the hits from the first and last broach at each surgery.
        :param p_early: precentage of final hits from the first broach (0.1, 0.2, 0.3 ...)
        :param p_late: precentage of final hits from the last broach (0.1, 0.2, 0.3 ...)
        :return: parsed and labeled dataframe
        '''

        indexs_early = []
        indexs_late = []


        for i in self.final_data['surgery number'].unique():

            d = self.final_data[self.final_data['surgery number'] == i]

            # early broach
            if d['broach index'].min() == 1:
                d_early = d[d['broach index'] == 1]

                for ind in d_early.iloc[:math.floor(p_early * d_early['number of hits'].iloc[0])]['index'].unique():
                    indexs_early.append(ind) #extract the final p% of the hits in the broach

            #late broach
            d_late = d[d['broach index'] == d['broach index'].max()]

            for ind in d_late.iloc[math.floor(-p_late * d_late['number of hits'].iloc[0]):]['index'].unique():
                indexs_late.append(ind) #extract the final p% of the hits in the broach


        print("early hits = ", str(len(indexs_early)))
        print("late hits = ", str(len(indexs_late)))
        merged_indexes = indexs_early + indexs_late
        parsed_data = self.final_data[self.final_data['index'].isin(merged_indexes)]
        parsed_data = parsed_data.dropna()

        def labeledParsedData(row):
            if (row['shaft size'] == 1):
                return 0  # early
            else:
                return 1  # late

        parsed_data['y'] = parsed_data.apply(lambda x: labeledParsedData(x), axis=1)

        return parsed_data

    def stat_tests(self, data, a, manual):
        '''
            Receives the data variable of type dataframe and performs a statistical test
            for each of the columns when the groups are late hits vs. early hits.
        :param data:
        :param manual:
        :return:Returns a dataframe only with the columns that passed the test with a certainty of over 95%.
        '''

        if manual:
            data.pop('index')
            data.pop('broach index')
            data.pop('hit index')

        else:
            data.pop('index')
            data.pop('surgeon name')
            data.pop('surgery index')
            data.pop('number of hits')
            data.pop('shaft size')
            data.pop('broach index')
            data.pop('date')
            # data.pop('Unnamed: 0')

        pass_t_test = []
        d1 = data.copy()
        f_n = d1.drop(['y' , 'surgery number'], axis = 1).columns.values.tolist()

        def two_side_mannwhitneyu(x, y, col_name, a):

            u, p_val = mannwhitneyu(x, y, alternative='two-sided')

            if p_val < a:
                pass_t_test.append(col_name)


        for i in range(len(f_n)):
            two_side_mannwhitneyu(data.loc[data['y'] == 0, data.columns == f_n[i]],
                                  data.loc[data['y'] == 1, data.columns == f_n[i]],  f_n[i], a)

        print("num remain:", len(pass_t_test))
        print("num start:", len(f_n))
        print("num pop:", len(f_n) - len(pass_t_test))
        pass_t_test.append('y')
        pass_t_test.append('surgery number')
        data = data[pass_t_test]
        data = self.clean_db(data)

        return data

    def realStartModelinPleaseStandUp(self, data1 , models):
        '''

        :param data: hits data
        :param models: array of model object with different hyperparameters in it!
        :return:
        '''

        data = data1.copy()

        #[9,12,20,30]
        test_surgeries = [9, 10, 20, 30] #There is no surgery 30 so we have 3 surgeries in the test
        test = data[data['surgery number'].isin(test_surgeries)]

        y_test = test['y']
        x_test = test.drop(columns=['y', 'surgery number'])
        x_test = preprocessing.normalize(x_test)

        print("1/0 ratio test - " + str(sum(y_test)/len(y_test)) + " test size:" + str(len(y_test)))


        data = data[~data['surgery number'].isin(test_surgeries)]

        #validation_surgeries = [ele for ele in data['surgery number'].unique() if ele not in test_surgeries]
        validation_surgeries = [ele for ele in data['surgery number'].unique()]
        validation_surgeries = shuffle(validation_surgeries, random_state=69)

        #list of lists
        validation_data = [validation_surgeries[i:i + 3] for i in range(0, len(validation_surgeries), 3)]

        for i in validation_data:
            if len(i) == 1:
                validation_data[0].append(i[0])
                validation_data.remove(i)

        res_train = {}
        res_val = {}
        res_test = {}
        fitted_models = []

        for m in models:

            res_train[m] = []
            res_val[m] = []
            res_test[m] = []

            for cv in range(len(validation_data)): #7 folds

               print("----- cv : " + str(cv + 1))
               print("----- validation surgeries : " + str(validation_data[cv]))

               train = data[~data['surgery number'].isin(validation_data[cv])]

               y_train = train['y']
               x_train = train.drop(columns=['y', 'surgery number'])
               x_train = preprocessing.normalize(x_train)

               validation = data[data['surgery number'].isin(validation_data[cv])]

               y_val = validation['y']
               x_val = validation.drop(columns=['y','surgery number'])
               x_val = preprocessing.normalize(x_val)


               print("------ train size = " + str(len(x_train)) + " validation size = " + str(len(x_val)))
               print("1/0 ratio vali - " + str(sum(y_val) / len(y_val)))
               print("1/0 ratio train - " + str(sum(y_train) / len(y_train)))

               clf = m
               clf = clf.fit(x_train, y_train)

               roc_train = roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1])
               roc_val = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
               roc_test = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])

               res_train[m].append(roc_train)
               res_val[m].append(roc_val)
               res_test[m].append(roc_test)
               fitted_models.append(clf)

               print("model " + str(clf) + " fold " + str(cv) + " train score: " + str(roc_train)
                     + " validation score: " + str(roc_val) + " test score: " + str(roc_test))

        # return res_train, res_val, fitted_models
        return res_train, res_val, fitted_models, y_test, x_test

    def checkHistograms(self, df, feature):

        # fig = plt.figure(figsize=(20, 10))
        df_early = df[df['y'] == 0]
        df_late = df[df['y'] == 1]

        if(len(feature) <= 4):
            feature_to_show = feature + "0 Hz"
        else:
            feature_to_show = feature


        plt.hist(df_early[feature],
                 label=f"early - $\mu= {df_early[feature].mean(): .1f}, \ \sigma= {df_early[feature].std(): .1f}$",
                 density=True,
                 alpha=0.75)
        plt.hist(df_late[feature],
                 label=f"late -$\mu= {df_late[feature].mean(): .1f}, \ \sigma= {df_late[feature].std(): .1f}$",
                 density=True,
                 alpha=0.75 )

        plt.title(("Distribution of feature - " + feature_to_show + " between the early hits and the late hits"), fontsize=20)
        plt.xlabel(feature_to_show, fontsize=16)
        plt.ylabel("Probability density", fontsize=16)

        plt.legend()
        plt.show()

    # def plotAuc(self, res_train, res_val, fitted_models, data):
    def plotAuc(self, res_train, res_val, fitted_models, y_test, x_test, data):

        val_scores_all = []

        for key in res_val.values():
            for val in key:
                val_scores_all.append(val)

        res_test = {}

        ms = []
        max = 0
        max_index = -1

        for index, model_score in enumerate(zip(fitted_models, val_scores_all)):

            if model_score[1] > max:
                max = model_score[1]
                max_index = index

            if (index+1) % 9 == 0:
                ms.append(fitted_models[max_index])
                max = 0
                max_index = 0

        for m in ms:

            roc_test = roc_auc_score(y_test, m.predict_proba(x_test)[:, 1])
            res_test[m]= roc_test

            cf_matrix = confusion_matrix(y_test, m.predict(x_test))

            group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

            group_counts = ["{0:0.0f}".format(value) for value in
                            cf_matrix.flatten()]

            group_percentages = ["{0:.2%}".format(value) for value in
                                 cf_matrix.flatten() / np.sum(cf_matrix)]

            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                      zip(group_names, group_counts, group_percentages)]

            labels = np.asarray(labels).reshape(2, 2)
            ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
            ax.set_title('Model ' + str(m) + 'Confusion Matrix with labels\n\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            # Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['False', 'True'])
            ax.yaxis.set_ticklabels(['False', 'True'])
            # Display the visualization of the Confusion Matrix.
            plt.show()

        #plotting the CV results

        for key in list(res_train.keys()):

            train_mean = round(np.mean(list(res_train[key])), 3)
            train_std = round(np.std(list(res_train[key])), 3)

            validation_mean = round(np.mean(list(res_val[key])), 3)
            validation_std = round(np.std(list(res_val[key])), 3)

            X = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9']
            #X = ['Fold 1 ', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
            train_scores = list(res_train[key])
            val_scores = list(res_val[key])

            X_axis = np.arange(len(X))
            bar1 = plt.bar(X_axis - 0.2, train_scores, 0.4, label='Train = ' + str(train_mean) + " +/- " + str(train_std))
            bar2 = plt.bar(X_axis + 0.2, val_scores, 0.4, label='Validation = ' + str(validation_mean) + " +/- " + str(validation_std))
            plt.xticks(X_axis, X)
            plt.xlabel("Fold number")
            plt.ylabel("AUC")
            plt.title("Model " + str(key) + "AUC score per fold")
            plt.legend()
            plt.show()



        #plotting the Test resulits
        models_used = list(range(len(res_test.keys())))
        ####models_used = [str(i) for i in ms]
        test_scores = list(res_test.values())

        bar1 = plt.bar(models_used, test_scores)
        plt.xticks(models_used, rotation = 90)
        plt.ylabel("AUC")
        plt.xlabel("Model number")
        plt.title("Test scores of the models")

        for rect in bar1:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')

        plt.show()



        #plotting features importance for the best model

        #for i,m in enumerate(fitted_models):
            #if i % 6 == 0:

        d1 = data.copy()
        d1 = d1.drop(columns=['y', 'surgery number'])

        for i, m in enumerate(ms):

            if str(m)[0] == 'L':

                feature_importance = pd.DataFrame(columns=['feature', 'importance'])
                importance = m.coef_[0]
                # importance = [abs(ele) for ele in importance]
                # summarize feature importance

                feature_importance['feature'] = list(d1.columns)
                feature_importance['importance'] = list(importance)
                feature_importance = feature_importance.sort_values(by='importance', ascending=False, key=abs)
                feature_importance_for_hist = feature_importance[:5]
                feature_importance = feature_importance[:10]  # Top 10 features

                # plot feature importance
                # plt.bar(feature_importance['feature'], feature_importance['importance'])
                p = ["orange" for i in range(len(feature_importance['feature']))]
                sns.barplot(data=feature_importance, x="feature", y="importance", palette=p)
                plt.title("feature importance for model " + str(m))
                plt.show()

            else:
                feature_importance = pd.DataFrame(columns=['feature', 'importance'])
                importance = m.feature_importances_
                # summarize feature importance

                feature_importance['feature'] = list(d1.columns)
                feature_importance['importance'] = list(importance)
                feature_importance = feature_importance.sort_values(by='importance', ascending=False)
                feature_importance_for_hist = feature_importance[:5]
                feature_importance = feature_importance[:10]  # Top 10 features

                # plot feature importance
                # plt.bar(feature_importance['feature'], feature_importance['importance'])
                p = ["orange" for i in range(len(feature_importance['feature']))]
                sns.barplot(data=feature_importance, x="feature", y="importance", palette = p)
                plt.title("feature importance for model " + str(m))
                plt.show()

            for i in feature_importance_for_hist['feature']:
                self.checkHistograms(data, str(i))

    def startModelin(self, data):


        surgery_index_unique = data['surgery number'].unique()

        p_model_score = {}
        fold = 5

        model = [LogisticRegression(), GradientBoostingClassifier(), XGBClassifier(),
                 DecisionTreeClassifier(), RandomForestClassifier()]

        # # [RandomForestClassifier(), DecisionTreeClassifier()]

        models_name = ["LogisticRegression", "GradientBoostingClassifier", "XGBClassifier",
                       "DecisionTreeClassifier", "RandomForestClassifier"]

        parameters_grid = [{'penalty': ['l2'], 'random_state': [42],
                             'solver': ['lbfgs'], 'max_iter': [15]},

                            # {"init": ["random", 'k-mean+s+'], "n_clusters": [2], "n_init": [20],# KMeans
                            #  "max_iter": [10], "random_state": [42]},

                           {"loss": ["exponential"], "learning_rate": [0.01], ## gradientboost
                            "min_samples_split":[5],
                            "min_samples_leaf": [10], "max_depth": [50],
                            "max_features": ["log2", "sqrt"], "criterion": ["friedman_mse"],
                            "subsample": [1.0], "n_estimators": [50]},

                           {'min_child_weight': [5], 'gamma': [0.5, 1.5], ## XGBOOST
                            'subsample': [1.0], 'colsample_bytree': [1.0],
                            'max_depth': [50]},

                           {'criterion': ['gini'], 'max_leaf_nodes': [2], ##DECISIONTREE
                            'min_samples_split': [5], "max_depth": [50],
                            'min_samples_leaf': [10]},

                           {"n_estimators": [50], "max_depth": [50], ## randomforest
                            "min_samples_split": [5], "min_samples_leaf": [10]}]

                           # {'base_estimator__max_depth': [i for i in range(2, 11, 2)], ## ADABOOST
                           #  'base_estimator__min_samples_leaf': [5, 10], 'n_estimators': [10, 50, 250, 1000],
                           #  'learning_rate': [0.01, 0.1]},
                           #  {"n_estimators": [10], "max_depth": [100],  ## randomforest
                           #   "min_samples_split": [2, 10], "min_samples_leaf": [1, 10]},
                           #
                           #  {'criterion': ['gini'], 'max_leaf_nodes': [10],  ##DECISIONTREE
                           #   'min_samples_split': [2, 10],
                           #   "max_depth": [100], 'min_samples_leaf': [1, 10]}]
        flag = True
        size = 0
        while flag or (size < 30):
            print("checking............")
            flag = True
            test_surgery_index = np.random.choice(surgery_index_unique, size=3)
            # print(f"{test_surgery_index}")
            if test_surgery_index[0] != test_surgery_index[1] and test_surgery_index[2] != test_surgery_index[1]:

                X_TEST = data[(data['surgery number'] == test_surgery_index[0]) |
                              (data['surgery number'] == test_surgery_index[1]) |
                              (data['surgery number'] == test_surgery_index[2])]
                Y_TEST = X_TEST['y']

                size = len(Y_TEST)
                X_TRAIN = data[~((data['surgery number'] == test_surgery_index[0]) |
                                 (data['surgery number'] == test_surgery_index[1]) |
                                 (data['surgery number'] == test_surgery_index[2]))]
                Y_TRAIN = X_TRAIN['y']

                ratio_test_0 = np.bincount(Y_TEST)[0] / len(Y_TEST)
                ratio_train_0 = np.bincount(Y_TRAIN)[0] / len(Y_TRAIN)
                ratio_train_test_0 = len(Y_TEST) / (len(Y_TEST) + len(Y_TRAIN))
                if 0.6 > ratio_test_0 > 0.4 and 0.6 > ratio_train_0 > 0.4:
                    flag = False
        print("disco disco good good.............")
        print("validation surgries:", test_surgery_index[0], test_surgery_index[1], test_surgery_index[2])
        print(f"Train size: {Y_TRAIN.shape}, Test size: {Y_TEST.shape}")

        for ii in [X_TRAIN, X_TEST]:
            ii.pop('y')
            ii.pop('surgery number')
        X_TRAIN = preprocessing.normalize(X_TRAIN)
        X_TEST = preprocessing.normalize(X_TEST)

        for i in range(1, fold + 1):
            data = shuffle(data, random_state=42)

            for m in range(len(model)):
                accurcies_per_fold = []
                auc_per_fold = []
                cv_res = []
                print(f'start modeling {models_name[m]}')

                # never_again = []
                    # flag = True
                    # size = 0
                    # while flag or (size < 30):
                    #     print("checking............")
                    #     flag = True
                    #     test_surgery_index = np.random.choice(surgery_index_unique, size=3)
                    #     # print(f"{test_surgery_index}")
                    #     if test_surgery_index[0] != test_surgery_index[1] and test_surgery_index[2] != test_surgery_index[1]\
                    #             and test_surgery_index[0] not in never_again and test_surgery_index[1] not in never_again \
                    #             and test_surgery_index[2] not in never_again:
                    #
                    #         X_TEST = data[(data['surgery number'] == test_surgery_index[0]) |
                    #                       (data['surgery number'] == test_surgery_index[1]) |
                    #                       (data['surgery number'] == test_surgery_index[2])]
                    #         Y_TEST = X_TEST['y']
                    #
                    #         size = len(Y_TEST)
                    #         X_TRAIN = data[~((data['surgery number'] == test_surgery_index[0]) |
                    #                          (data['surgery number'] == test_surgery_index[1]) |
                    #                          (data['surgery number'] == test_surgery_index[2]))]
                    #         Y_TRAIN = X_TRAIN['y']
                    #
                    #         ratio_test_0 = np.bincount(Y_TEST)[0] / len(Y_TEST)
                    #         ratio_train_0 = np.bincount(Y_TRAIN)[0] / len(Y_TRAIN)
                    #         ratio_train_test_0 = len(Y_TEST)/ (len(Y_TEST) + len(Y_TRAIN))
                    #         if 0.6 > ratio_test_0 > 0.4 and 0.6 > ratio_train_0 > 0.4:
                    #             flag = False
                    # print("disco disco good good.............")
                    # never_again.append(test_surgery_index[0])
                    # never_again.append(test_surgery_index[1])
                    # never_again.append(test_surgery_index[2])

                print(f"run GSCV for: {models_name[m]} , fold - {i}")
                # X_TEST, Y_TEST = shuffle(X_TEST, Y_TEST, random_state=42)
                #
                # X_TRAIN, Y_TRAIN = shuffle(X_TRAIN, Y_TRAIN, random_state=42)

                clf = GridSearchCV(cv=10,
                                   estimator=model[m],
                                   param_grid=parameters_grid[m],
                                   refit=True, return_train_score=True,
                                   scoring='roc_auc', verbose=1)


                clf = clf.fit(X_TRAIN, y=Y_TRAIN)
                # Predict the response for test dataset
                # clf = clf.evaluate(X_TEST.iloc[:30, :], Y_TEST.iloc[:30, :])

                y_pred = clf.predict(X_TEST)
                print('best param: ', clf.best_params_)

                # Model Accuracy, how often is the classifier correct?
                print("Accuracy:", metrics.accuracy_score(Y_TEST, y_pred))

                # # DOT data
                # dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                #                                 feature_names=list(X_TRAIN.columns),
                #                                 filled=True)
                print(metrics.classification_report(Y_TEST, y_pred))
                auc = roc_auc_score(Y_TEST, y_pred)
                accuracy = metrics.accuracy_score(Y_TEST, y_pred)
                accurcies_per_fold = np.append(accurcies_per_fold, accuracy)
                auc_per_fold = np.append(auc_per_fold, auc)
                cv_res = np.append(cv_res, clf.cv_results_)
            p_model_score[f'model-{models_name[m]}'] = [accurcies_per_fold, auc_per_fold, cv_res]
            # Draw graph
            # graph = graphviz.Source(dot_data, format="png", filename='pngg.png')
            # text_representation = tree.export_text(clf.best_estimator_, feature_names=list(data.columns))
            # print(text_representation)
            # fig = plt.figure(figsize=(20, 20))
            # _ = tree.plot_tree(clf.best_estimator_,
            #                    feature_names=list(data.columns),
            #                    class_names=['0', '1'],
            #                    filled=True,
            #                    fontsize=5, rounded=True)
            # # plt.show()

            # importances = clf.best_estimator_.feature_importances_
            # sorted_indices = np.argsort(importances)[::-1][:10]
            # plt.title('Feature Importance')
            # plt.bar(range(len(sorted_indices)), importances[sorted_indices], align='center')
            # plt.xticks(range(len(sorted_indices)), X_TRAIN.columns[sorted_indices], rotation=90)
            # plt.tight_layout()
            # # plt.show()

        return p_model_score

    def plotAcc(self, dict1, p, txt):

        X = ['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5']
        X_axis = np.arange(len(X))
        best_model_score = []
        i = 0

        for m in dict1[1].keys():
            accuracys = dict1[p][m][0]
            aucs = dict1[p][m][1]

            mean_acc = round(np.mean(dict1[p][m][1]), 2)
            std_acc = round(np.std(dict1[p][m][1]), 2)

            mean_auc = round(np.mean(dict1[p][m][0]), 2)
            std_auc = round(np.std(dict1[p][m][0]), 2)
            best_model_score.append([m, mean_auc, std_auc])


            plt.bar(X_axis - 0.2, accuracys, 0.4, label='Mean accuracy {} +- {}'.format(mean_acc, std_acc))
            # plt.axhline(mean_acc, color='black', ls='dotted')

            plt.bar(X_axis + 0.2, aucs, 0.4, label='Mean auc {} +- {}'.format(best_model_score[i][1], best_model_score[i][2]))
            plt.axhline(mean_auc, color='red', ls='dotted')
            plt.xlabel("Fold")
            plt.ylabel("%")
            plt.legend()

            plt.xticks(X_axis, X)
            plt.title(f"{txt} - Average (AUC/Accuracy) - {best_model_score[i][0].split('-')[1]}")
            plt.show()
            i +=1

        X = ["LogisticRegression", "KMeans", "GradientBoostingClassifier", "XGBClassifier",
                       "DecisionTreeClassifier", "RandomForestClassifier"]
        X_axis = np.arange(len(X))

        for i in range(len(X)):
            plt.bar(i, best_model_score[i][1], label=f"{best_model_score[i][0].split('-')[1]}-{best_model_score[i][1]}+-{best_model_score[i][2]}")
            plt.xlabel("Models")
            plt.ylabel("%")
            plt.legend()

        plt.xticks(X_axis, X)
        plt.title(f"{txt} - Average (AUC/Accuracy) all the models")
        plt.show()

            # plt.bar(x=range(1, 3), height=accuracys)
            # plt.bar(x=range(1, 3), height=aucs)
            #
            # plt.title("average auc {} +/- {} , average accuracy {} +/- {}"
            #           .format(round(np.mean(dict1[i / 10][1]), 2), round(np.std(dict1[i / 10][1]), 2),
            #                   round(np.mean(dict1[i / 10][0]), 2), round(np.std(dict1[i / 10][0]), 2)))
            # # plt.show()

            # plt.xticks()
        # plt.title("Accuracy per final hits percentage")
        # plt.legend()
        # plt.ylim(min(dict1.values()), max(dict1.values()))
        # plt.show()

    def clean_db(self, X):
        # drop 0 variance columns:
        del_col = []
        for i in X.columns[:-1]:
            if np.std(X.loc[:, i]) < 0.01 * np.mean(X.loc[:, i]):
                del_col.append(i)
        print(del_col, 'have 1% variance, will be dropped from db')
        X = X.drop(del_col, axis=1)

        # drop 90% correlated columns
        corr_db = pd.DataFrame(np.corrcoef(X.transpose().astype(float)),
                               index=X.columns,
                               columns=X.columns)
        del_col = []
        for c_index, c in enumerate(corr_db.columns):
            for b in corr_db.index[c_index + 1:]:
                if corr_db.loc[c, b] >= 0.9 and b != c:
                    # print(c, ' and ', b, ' are strongly associated: ', corr_db.loc[c, b])
                    if b not in del_col:
                        del_col.append(b)
        print("deleting column ", del_col)
        print("Total deleted columns = ", len(del_col))
        X = X.drop(del_col, axis=1)
        #
        # LEN_X = X.shape
        #
        # for col in X.columns:
        #     std_ = np.std(X[col])
        #     mean_ = np.mean(X[col])
        #     X = X[X[col] < (mean_ + 1.5*std_)]
        #     X = X[X[col] > (mean_ - 3*std_)]
        # lenx = X.shape
        #
        # print(f"original data: {LEN_X[0]} rows, {LEN_X[1]} col:  new data: {lenx[0]} rows, {lenx[1]} col: ")

        """
        # z-score normalization - not meant for regressions!
        for i in X.columns:
            X.loc[:,i] = (X.loc[:,i]-np.mean(X.loc[:,i]))/np.std(X.loc[:,i])
        """
        return X

    def cv_split_by_surgery(self, X, Y, surgery_index):  # leave surgery out using 'surgery_index' vector
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        surgery_set = {}
        for i in list(set(surgery_index)):
            surgery_set[i] = Y.iloc[np.where(surgery_index == i)].values[0]
        # K-fold Cross Validation model evaluation
        fold_no = 1
        x_train = {}
        y_train = {}
        x_test = {}
        y_test = {}
        for train, test in kfold.split(list(surgery_set.keys()), list(surgery_set.values())):
            train_index = list(np.where([i in train for i in surgery_index])[0])
            test_index = list(np.where([i in test for i in surgery_index])[0])
            x_train[fold_no] = X.iloc[train_index, :].astype(float)
            y_train[fold_no] = Y.iloc[train_index].astype(int)
            x_test[fold_no] = X.iloc[test_index, :].astype(float)
            y_test[fold_no] = Y.iloc[test_index].astype(int)
            fold_no = fold_no + 1
        return x_train, y_train, x_test, y_test

    def train_test(self, x_train, y_train, x_test, y_test):
        scores = {}
        feature_probs = pd.Series(np.zeros([len(x_train[1].columns)]), index=x_train[1].columns).astype(int)
        for i in x_train.keys():
            model = Logit(y_train[i], x_train[i])
            res = model.fit_regularized(method='l1', alpha=1)
            print(f"fold {i} res1:\n", res.summary())

            # alternatively, without summary implementation
            # res2 = sm.GLM(y_train[i],x_train[i], family=sm.families.Binomial()).fit_regularized(L1_wt=0.0, alpha=0.1)
            # print(f"fold {i} res2:\n" , res2.params)

            ext_features = x_train[i].columns[res.pvalues < (0.05 / len(res.pvalues))]
            feature_probs[ext_features] = feature_probs[ext_features] + 1
            # test model
            y_hat = res.predict(x_test[i])
            # save model performance
            scores[i] = {'roc auc': roc_auc_score(y_test[i], y_hat),
                         'confusion_matrix': confusion_matrix(y_test[i], round(y_hat).astype(int))}
        mean_roc_auc = np.mean([i['roc auc'] for i in scores.values()])
        std_roc_auc = np.std([i['roc auc'] for i in scores.values()])
        print("mean roc auc performance:\n", mean_roc_auc, "+/-", std_roc_auc)
        return feature_probs, scores

    def Logit_model(self, data):

        surgery_index_unique = data['surgery number'].unique()

        y = data['y']
        data.pop('y')
        data.pop('index')
        data.pop('surgeon name')
        data.pop('surgery index')
        data.pop('date')
        data.pop('surgery number')
        data.pop('number of hits')
        data.pop('shaft size')
        data.pop('broach index')

        X = data.iloc[:, 1200:1300]
        X = self.clean_db(X)

        x_train, y_train, x_test, y_test = self.cv_split_by_surgery(X=X, Y=y,
                                                                    surgery_index=surgery_index_unique)
        print('x_train =', len(x_train), 'y_train =', len(y_train),
              'x_test =', len(x_test), 'y_test =', len(y_test))

        feature_probs, scores = self.train_test(x_train, y_train, x_test, y_test)
        X_new = X.loc[:, feature_probs >= 4]
        return X_new

    def plot_means(self, accurcies_per_fold, auc_per_fold):
        mean_accuracy_auc = np.mean(accurcies_per_fold)
        std_accuracy_auc = np.std(accurcies_per_fold)
        mean_roc_auc = np.mean(auc_per_fold)
        std_roc_auc = np.std(auc_per_fold)

    def PCA_TEST(self, data):
        data.pop('index')
        data.pop('surgeon name')
        data.pop('surgery index')
        data.pop('number of hits')
        data.pop('shaft size')
        data.pop('broach index')
        data.pop('date')

        data = self.clean_db(data)
        y = data['y']
        X = data.drop('y', axis=1)

        # lda = LDA(n_components=2)
        # X_train = lda.fit_transform(X_train, y_train)
        # len(X_train)
        # X_test = lda.transform(X_test)

        features = X.columns
        # PCA 3D

        pca = PCA(n_components=3)

        components = pca.fit_transform(X[features])

        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=y,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()

        # PCA 2D
        # features = X.columns
        #
        # pca = PCA()
        # components = pca.fit_transform(X[features])
        # labels = {
        #     str(i): f"PC {i + 1} ({var:.1f}%)"
        #     for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        # }
        #
        # fig = px.scatter_matrix(
        #     components,
        #     labels=labels,
        #     dimensions=range(3),
        #     color=y
        # )
        # fig.update_traces(diagonal_visible=True)
        # fig.show()


'''
dot_data = tree.export_graphviz(clf.best_estimator_[0], out_file=None,
                                                feature_names=list(X_TRAIN.columns),
                                                filled=True)
                graph = graphviz.Source(dot_data, format="png", filename='pngg.png')
                # text_representation = tree.export_text(clf.best_estimator_[0], feature_names=list(X_TRAIN.columns))
                # print(text_representation)
                fig = plt.figure(figsize=(20, 20))
                _ = tree.plot_tree(clf.best_estimator_[0],
                                   feature_names=colnames2,
                                   class_names=['0', '1'],
                                   filled=True,
                                   fontsize=5, rounded=True, max_depth=7)
                plt.show()

'''