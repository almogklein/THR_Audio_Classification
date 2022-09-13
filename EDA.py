import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EDA:
    def __init__(self, fft, features, data):
        self.fft = fft
        self.features = features
        self.data = data
        self.final_data = None

    def Merge(self, dict1, dict2):
        '''
        :param dict1:
        :param dict2:
        :return: merged dictionaries
        '''

        res = {**dict1, **dict2}
        return res

    def mergeDatasets(self):
        '''
        :param features: features CSV
        :param data: hits CSV
        :param fft : fft data
        :return: merged dataframe (all features in one dataframe)

        this function will rename and merge between the metadata, features and fft dataframes.
        '''

        d = self.data['1'].astype(str).str.split('', expand=True)


        self.data['broach index'] = d[2].astype(str) + d[3].astype(str)
        self.data['broach index'] = self.data['broach index'].astype(int)

        # zero_cross = {str(i) : "zero cross " + str(i) for i in range(8)}
        #logsval = {str(i): "log s value " + str(i - 6) for i in range(6, 134)}
        # mfcc = {str(i): "mfcc " + str(i - 134) for i in range(134, 154)}
        # chroma = {str(i): "chroma " + str(i - 154) for i in range(154, 167)}

        # logsval = {str(k) : 'log s val ' + str(j) + ' feature ' + str(i) for i in range(128) for j in range(8)}
        # mfcc = {str(j) + str(i): 'mfcc ' + str(j) + ' feature ' + str(i) for i in range(20) for j in range(8)}
        # mfcc_delta = {str(j) + str(i): 'mfcc_delta ' + str(j) + ' feature ' + str(i) for i in range(20) for j in range(8)}
        # mfcc_delta2 = {str(j) + str(i): 'mfcc_delta2 ' + str(j) + ' feature ' + str(i) for i in range(20) for j in range(8)}
        # chroma = {str(j) + str(i): 'chroma ' + str(j) + ' feature ' + str(i) for i in range(12) for j in range(8)}
        #
        # logsval = {}
        # c = 8
        # for i in range(128):
        #    logsval[str(c)] = 'log s val ' + str(i)
        #    c += 1
        #
        # mfcc = {}
        # for i in range(20):
        #     mfcc[str(c)] = 'mfcc ' + str(i)
        #     c += 1
        #
        # mfcc_delta = {}
        # for i in range(20):
        #     mfcc_delta[str(c)] = 'mfcc_delta ' + str(i)
        #     c += 1
        #
        # mfcc_delta2 = {}
        # for i in range(20):
        #     mfcc_delta2[str(c)] = 'mfcc_delta2 ' + str(i)
        #     c += 1
        #
        # chroma = {}
        # for i in range(12):
        #     chroma[str(c)] = 'chroma ' + str(i)
        #     c += 1

        # temp = {'Unnamed: 0': 'index'}
        # columns = self.Merge(zero_cross, logsval)
        # columns = self.Merge(columns, mfcc)
        # columns = self.Merge(columns, mfcc_delta)
        # columns = self.Merge(columns, mfcc_delta2)
        # columns = self.Merge(columns, chroma)
        # columns = self.Merge(columns, temp)



        # columns = zero_cross | logsval | mfcc| mfcc_delta | mfcc_delta2 | chroma | temp #works only with python 3.9+

        # self.features = self.features.rename(columns=columns)


        self.data = self.data.rename(columns={'Unnamed: 0': 'index'})
        self.features = self.features.rename(columns={'Unnamed: 0': 'index'})
        self.fft = self.fft.rename(columns={'Unnamed: 0': 'index'})

        self.data['index'] = self.data['index']

        meta_data = self.data.rename(
            columns={'0': 'surgeon name', '1': 'surgery index', '2': 'shaft size', '3': 'date',
                     '4': 'surgery number', '5': 'number of hits'})


        zero_cross = {str(i): "zero cross " + str(i) for i in range(8)}

        logsval = {}
        c = 8
        for i in range(128):
            logsval[str(c)] = 'log s val ' + str(i)
            c += 1

        mfcc = {}
        for i in range(20):
            mfcc[str(c)] = 'mfcc ' + str(i)
            c += 1

        mfcc_delta = {}
        for i in range(20):
            mfcc_delta[str(c)] = 'mfcc_delta ' + str(i)
            c += 1

        mfcc_delta2 = {}
        for i in range(20):
            mfcc_delta2[str(c)] = 'mfcc_delta2 ' + str(i)
            c += 1

        chroma = {}
        for i in range(12):
            chroma[str(c)] = 'chroma ' + str(i)
            c += 1

        temp = {'208': 'temp', 'Unnamed: 0': 'index'}
        columns = self.Merge(zero_cross, logsval)
        columns = self.Merge(columns, mfcc)
        columns = self.Merge(columns, mfcc_delta)
        columns = self.Merge(columns, mfcc_delta2)
        columns = self.Merge(columns, chroma)
        columns = self.Merge(columns, temp)

        self.features = self.features.rename(columns=columns)

        self.final_data = pd.merge(meta_data, self.features,
                              on='index',
                              how='right')

        self.final_data = pd.merge(self.final_data, self.fft,
                                   on = 'index',
                                   how='right')

        return self.final_data

    def ShowHitsPerSurgery(self):

        hits = {i : 0 for i in range(1,33)} #dict of surgery number and number of hits - initialize with 0

        runable_data = self.final_data.copy()
        runable_data['run'] = runable_data['surgery index'].astype(str) + runable_data['date'].astype(str) + runable_data['surgery number'].astype(str)

        for i in runable_data['run'].unique():
            d = runable_data[runable_data['run'] == i]
            hits[d['surgery number'].iloc[0]] += (d['number of hits'].sum() / len(d['number of hits']))

        #print(hits)
        sns.barplot( x = list(hits.keys()) , y = list(hits.values()) , palette="Blues_d").set(title = 'Hits per surgery' ,
                                                                          xlabel = 'Surgery number', ylabel = 'Number of hits')
        plt.show()

    def ShowHitsPerSurgeon(self):
        surgeon_hits = {'RO' : 0 , 'OM' : 0 , 'AM' : 0}
        surgeon_surgeries = {'RO' : 0 , 'OM' : 0 , 'AM' : 0}
        Avg_hits_per_surgeon = {'RO' : 0 , 'OM' : 0 , 'AM' : 0}

        runable_data = self.final_data.copy()

        for i in runable_data['surgeon name'].unique():

            d = runable_data[runable_data['surgeon name'] == i]
            surgeon_surgeries[i] = d['surgery number'].nunique()
            d['run'] = d['surgery index'].astype(str) + d['date'].astype(str) + d['surgery number'].astype(str)

            for j in d['run'].unique():
                k = d[d['run'] == j]
                surgeon_hits[i] += (k['number of hits'].sum() / len(k['number of hits']))

        for key in surgeon_hits:
            if key in surgeon_surgeries:
                Avg_hits_per_surgeon[key] = surgeon_hits[key] / surgeon_surgeries[key]
            else:
                pass

        sns.barplot( x = list(surgeon_surgeries.keys()) , y = list(surgeon_surgeries.values())).set(title = 'Surgeries per surgeon' ,
                                                                                          xlabel = 'Surgeon name', ylabel = 'Number of surgeries')
        plt.show()

        sns.barplot( x = list(surgeon_hits.keys()) , y = list(surgeon_hits.values())).set(title = 'Hits per surgeon' ,
                                                                           xlabel = 'Surgeon name', ylabel = 'Number of hits')
        plt.show()

        sns.barplot( x = list(Avg_hits_per_surgeon.keys()) , y = list(Avg_hits_per_surgeon.values())).set(title = 'AVG Hits per surgeon' ,
                                                                                          xlabel = 'Surgeon name', ylabel = 'AVG Number of hits')
        plt.show()

    def ShowHitsInSurgeryByShaft(self):


        runable_data = self.final_data.copy()

        # fig, axes = plt.subplots(7 , 5 , figsize = (25,30))
        # fig.suptitle('Hits per broach in each surgery')
        # c_i = 0
        # c_j = 0

        for i in runable_data['surgery number'].unique():

            d = runable_data[runable_data['surgery number'] == i]
            d['y'] = [10 for i in d['index']]
            #ax = axes[c_j, c_i]


            sns.barplot( data = d ,x = 'index' , y = 'y' , hue = 'shaft size').set(title = 'Hits for surgery ' + str(i) ,
                                                                                                                       xlabel = 'Hit', ylabel = 'DOESNT METTER')

            plt.legend(loc='best')
            plt.savefig(str(i) +'.png')
            plt.show()
            # c_i += 1
            # if(c_i % 5 == 0):
            #     c_i = 0
            #     c_j += 1

        # plt.legend(loc='best')
        #

    def ShowHitStatistics(self):

        runable_data = self.final_data.copy()

        fig, axes = plt.subplots(7, 5, figsize = (25,30))
        fig.suptitle('Hits per shaft in each surgery')
        c_i = 0
        c_j = 0
        for i in runable_data['surgery number'].unique():

            d = runable_data[runable_data['surgery number'] == i]
            dict1 = { j : len(d[d['shaft size'] == j]) for j in d['shaft size'].unique() }

            sns.barplot(ax = axes[c_j, c_i], x=list(dict1.keys()), y=list(dict1.values())).set(
                title='Hits per shaft for surgery ' + str(i),
                xlabel='Shaft size', ylabel='Number of hits')
            c_i += 1
            if(c_i % 5 == 0):
                c_i = 0
                c_j += 1
            print(str(c_j) + "," + str(c_i))


        plt.show()

    def ShowMoreStatistics(self):

        runable_data = self.final_data.copy()

        fig, axes = plt.subplots(7 , 5 , figsize = (25,30))
        fig.suptitle('Hits per broach in each surgery')
        c_i = 0
        c_j = 0
        for i in runable_data['surgery number'].unique():

            d = runable_data[runable_data['surgery number'] == i]
            dict1 = { j : len(d[d['broach index'] == j]) for j in d['broach index'].unique() }


            sns.barplot(ax = axes[c_j, c_i], x=list(dict1.keys()), y=list(dict1.values())).set(
                title='Hits per broach for surgery ' + str(i),
                xlabel='Shaft size', ylabel='Number of hits')
            c_i += 1
            if(c_i % 5 == 0):
                c_i = 0
                c_j += 1
            #print(str(c_j) + "," + str(c_i))


        plt.show()

    def ShowShaftPrecentFirstBroach(self):

        runable_data = self.final_data.copy()

        d = runable_data[runable_data['broach index'] == 1]

        print(d)

        dict1 = {i : len(d[d['shaft size'] == i])/len(d) * 100 for i in d['shaft size'].unique()}

        sns.barplot(x=list(dict1.keys()), y=list(dict1.values()), palette="Blues_d").set(title='Shaft size precentage for the third branch',
                                                                                       xlabel='Shaft size',
                                                                                       ylabel='%')
        plt.show()

    def ShowShaftPrecentLastBroach(self):

        runable_data = self.final_data.copy()
        dict1 = {}

        for i in runable_data['surgery number'].unique():

            n_d = runable_data[runable_data['surgery number'] == i]
            n_d_max = n_d[n_d['broach index'] == n_d['broach index'].max() - 2]['shaft size'].iloc[0] #shaft size in final broach number

            if(n_d_max in dict1.keys()):
                dict1[n_d_max] = dict1[n_d_max] + len(n_d)
            else:
                dict1[n_d_max] = len(n_d)

        sum1 = sum(dict1.values())
        for k in dict1.keys():
            dict1[k] = (dict1[k]/sum1) * 100

        sns.barplot(x=list(dict1.keys()), y=list(dict1.values()), palette="Blues_d").set(title='Shaft size precentage for the two before last branch',
                                                                                       xlabel='Shaft size',
                                                                                       ylabel='%')
        plt.show()


