import numpy as np
import pandas as pd
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
from scipy.io.wavfile import write
import re


class Extract_Features():

    def align_data(self, data, sr):
        delta = int(sr * 0.005)
        max_value = max(data)
        peak_index = np.where(data == max_value)[0][0]
        if peak_index > delta:
            data = data[peak_index - delta:]
        else:
            return data

        return data

    # # find difference between final hits of early broach to final hits late bro
    def load_metadata_database(self, path, CSV_READ, PLOT, sample_rate,
                               HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER):

        FORWARD_WINDOW_LENGTH_OF_PEAK = int(0.17 / (1 / sample_rate))  # 100 milisec window / (1 / sample rate)
        BACKWARD_WINDOW_LENGTH_OF_PEAK = int(0.03 / (1 / sample_rate))  # 30 milisec window / (1 / sample rate)
        NO_BACKWARD_WINDOW = 0
        surgery_index = 0
        hits_dict = {}
        features_dict_Backward_Forward = {}
        FFT_dict = {}
        window_length = 6400
        WINDOW_COUNTER = 0

        if CSV_READ:
            for i, file_name in enumerate(glob.glob(os.path.join(path, '*.wav'))):

                data, sr = librosa.load(file_name, sr=sample_rate)
                D = np.abs(librosa.stft(data, n_fft=1024, hop_length=512, window="hann"))
                onset_env = librosa.onset.onset_strength(y=data, sr=sr, aggregate=np.max, S=D)
                times = librosa.times_like(onset_env, sr=sr, n_fft=1024, hop_length=512)
                onset_frames = librosa.onset.onset_detect(y=data, onset_envelope=onset_env, sr=sr, normalize=True)

                if PLOT:
                    self.plot_graph_time_freq_peak(D, data, sr, times, onset_env, onset_frames)

                meta_data = file_name.split('\\')[5]
                file_id = meta_data.split('-')[0]
                date = meta_data.split('-')[1]
                surgon_name = meta_data.split('-')[2]
                broach_index = meta_data.split('-')[3]
                relative_shaft_size = meta_data.split('-')[4].split('.')[0]

                total_hits_detected_in_broach = len(times[onset_frames])
                print("total_hits_detected_in_broach:", total_hits_detected_in_broach)

                if 'stem' in relative_shaft_size.lower():
                    print("deleted one broach with 'stem'")
                    surgery_index += 1
                    print(f"end of surgery {surgery_index}")
                    continue
                elif 'cerclage' in relative_shaft_size.lower():
                    print("deleted one broach with 'cerclage'")
                    continue
                else:
                    for hit_index, time_index in enumerate(times[onset_frames]):

                        center_of_peak_in_time = int(time_index * sr)
                        bottom_of_peak_in_time = int(center_of_peak_in_time - BACKWARD_WINDOW_LENGTH_OF_PEAK)
                        top_of_peak_in_time = int(center_of_peak_in_time + FORWARD_WINDOW_LENGTH_OF_PEAK)

                        dictionary_index = "broach_{} hit_index_{}".format(file_id, hit_index)

                        hits_dict[dictionary_index] = np.array([surgon_name, broach_index,
                                                                relative_shaft_size, date, surgery_index,
                                                                total_hits_detected_in_broach])
                        if len(data[bottom_of_peak_in_time:top_of_peak_in_time]) > 0:
                            #
                            # if len(data) > window_length * 10:
                            #     print("wrong windowing")
                            #     WINDOW_COUNTER += 1
                            #     continue
                            # if len(data) > window_length:
                            #     data = data[:window_length]
                            # if len(data) < window_length:
                            #     data = np.insert(data, len(data), np.zeros((window_length - len(data),)))

                            total_values_Backward_Forward = self.extract_features(data[bottom_of_peak_in_time:top_of_peak_in_time], sr,
                                                            FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK,
                                                            HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)

                           # # total_values_Forward = self.extract_features(data[center_of_peak_in_time:top_of_peak_in_time], sr,
                           #  #                                   FORWARD_WINDOW_LENGTH_OF_PEAK, NO_BACKWARD_WINDOW,
                           #   #                                  HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)
                           #
                            write(rf"C:\Users\python3\PycharmProjects\pythonProject9\check\{dictionary_index}.wav", sr,
                                  data[bottom_of_peak_in_time:top_of_peak_in_time])

                            #write(rf"C:\Users\python3\PycharmProjects\pythonProject9\check\F_{dictionary_index}.wav", sr,
                             #     data[center_of_peak_in_time:top_of_peak_in_time])
                            features_dict_Backward_Forward[dictionary_index] = total_values_Backward_Forward

                            FFT_dict[dictionary_index] = self.plot_fft(data[bottom_of_peak_in_time:top_of_peak_in_time], sr)

            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hits_dict.items()]))
            df = df.transpose()
            df.to_csv("DATA.csv")
            print('-----------------------------DATA.csv created-----------------------------')

            df_features_Backward_Forward = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict_Backward_Forward.items()]))
            df_features_Backward_Forward = df_features_Backward_Forward.transpose()
            df_features_Backward_Forward.to_csv('features_data_.csv')
            print('-----------------------------features_data_.csv created---------------------------')
            print(f'row: {df_features_Backward_Forward.shape[0]}, features: {df_features_Backward_Forward.shape[1]}')

            df_features_FFT = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in FFT_dict.items()]))
            df_features_FFT = df_features_FFT.transpose()
            df_features_FFT.to_csv('features_data_FFT.csv')
            print('-----------------------------features_data_FFT.csv created-----------------------------')
            print(f'row: {df_features_FFT.shape[0]}, features: {df_features_FFT.shape[1]}')

    def extract_features(self, data, sr, FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK,
                         HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER):

        if len(data) > 0:
            zero_cross = librosa.feature.zero_crossing_rate(
                y=data, frame_length=int(len(data) / N_FFT_DIVISION_NUMBER),
                hop_length=int(
                    (FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK) / HOP_LENGTH_DIVISION_NUMBER))
            zero_cross = np.reshape(zero_cross, np.shape(zero_cross)[1])

            S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                               hop_length=int(
                                                   (FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                   / HOP_LENGTH_DIVISION_NUMBER))
            log_s = librosa.power_to_db(S, ref=np.max)
            log_s_values = np.max(log_s, axis=1)

            mfcc1 = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                         hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                        / HOP_LENGTH_DIVISION_NUMBER))
            mfcc_values = np.max(mfcc1, axis=1)

            mfcc_delta = librosa.feature.delta(mfcc1, width=3, order=1, axis=0)
            mfcc_delta_values = np.max(mfcc_delta, axis=1)

            mfcc_delta2 = librosa.feature.delta(mfcc1, width=3, order=2, axis=0)
            mfcc_delta2_values = np.max(mfcc_delta2, axis=1)

            chromagram = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                                     hop_length=int(
                                                         (FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                         / HOP_LENGTH_DIVISION_NUMBER))
            chromagram_values = np.max(chromagram, axis=1)

            onset_env = librosa.onset.onset_strength(y=data, sr=sr)
            tempo = librosa.beat.tempo(y=onset_env, sr=sr)

            total_values = np.concatenate((zero_cross, log_s_values, mfcc_values, mfcc_delta_values, mfcc_delta2_values,
                                           chromagram_values, tempo))
        else:
            total_values = None

        return total_values

    def plot_graph_time_freq_peak(self, D, data, sr, times, onset_env, onset_frames):

        fig, ax = plt.subplots(3, 1, constrained_layout=True)

        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                 x_axis='time', y_axis='log', ax=ax[0])
        ax[0].set(title='Power spectrogram (time-frequency)')
        ax[0].set(xlabel='time')
        ax[0].label_outer()

        librosa.display.waveshow(data, sr=sr, ax=ax[1])
        ax[1].set(ylabel='Amplitude')
        ax[1].set(xlabel='time')
        ax[1].set(title='Raw signal (time)')

        ax[2].plot(times, onset_env, label='Onset strength')
        ax[2].vlines(times[onset_frames], 0, onset_env.max(), color='r', alpha=0.9,
                     linestyle='--', label='Onsets')
        ax[2].legend()
        ax[2].set(title='Smooth signal (time)')
        ax[2].set(ylabel='Amplitude (abs)')
        ax[2].set(xlabel='time')

        plt.show()

    def check_hits(self, path, sample_rate, HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER):

        hits_dict = {}
        features_dict = {}
        FORWARD_WINDOW_LENGTH_OF_PEAK = 0.05 * sample_rate
        BACKWARD_WINDOW_LENGTH_OF_PEAK = 0.05 * sample_rate

        for i, file_name in enumerate(glob.glob(os.path.join(path, '*.wav'))):

            data, sr = librosa.load(file_name, sr=sample_rate)

            file_id = file_name.split('\\')[5]
            surgery_index = re.split('(\d+)', file_id)[1]
            broach_index = re.split('(\d+)', file_id)[3]
            hit_index = re.split('(\d+)', file_id)[5]

            dictionary_index = "surgery_{}_broach_{}_hit_index_{}".format(surgery_index, broach_index, hit_index)

            hits_dict[dictionary_index] = np.array([surgery_index, broach_index, hit_index])

            total_values = self.extract_features(data, sr, FORWARD_WINDOW_LENGTH_OF_PEAK,
                                                                  BACKWARD_WINDOW_LENGTH_OF_PEAK,
                                                                  HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)
            features_dict[dictionary_index] = total_values
            print(f'features: {total_values.shape}')

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hits_dict.items()]))
        df = df.transpose()
        df.to_csv("DATA_manual.csv")
        print('--------------***---------------DATA_manual.csv created-----------------***------------')

        df_features = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict.items()]))
        df_features = df_features.transpose()
        df_features.to_csv('features_data_manual.csv')
        print('----------------***-------------features_data_manual.csv created-----------***----------------')
        print(f'row: {df_features.shape[0]}, features: {df_features.shape[1]}')

    def plot_fft(self, signal, sr):
        if len(signal) > 0:
            fft_spectrum = np.fft.rfft(signal)
            fft_spectrum_abs = np.abs(fft_spectrum)
            freq = np.fft.rfftfreq(signal.size, d=1. / sr)
            # plt.plot(freq, fft_spectrum_abs, label=str)
            # plt.xlabel("frequency, Hz")
            # plt.ylabel("Amplitude, units")
            # plt.title(str)
            # plt.legend()
            # plt.show()
        else:
            fft_spectrum_abs = None
        return fft_spectrum_abs
#
# if __name__ == '__main__':
#
#     path = r"C:\Users\almog\data_thr\raw data"
#     CSV_READ = True
#     PLOT = False
#     sample_rate = 64000
#     HOP_LENGTH_DIVISION_NUMBER = 7
#     N_FFT_DIVISION_NUMBER = 3.5
#
#     Extract_Features.load_metadata_database(path, CSV_READ, PLOT, sample_rate,
#                                             HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)

# # from pyAudioAnalysis import audioBasicIO
# # from pyAudioAnalysis import ShortTermFeatures
# # from pydub import AudioSegment
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (20, 10)
# import tqdm
# import glob
# import os
# import librosa
# import time
# from librosa import display
# import pandas as pd
# from scipy.fftpack import fft
# from scipy.signal import periodogram
# import seaborn as sns
# from scipy.io import wavfile
#
# path = r"C:\Users\DELL\audiodata"
# CSV_READ = False
# FORWARD_WINDOW_LENGTH_OF_PEAK = int(0.12 / (1 / 64000))  # 120 milisec window / (1 / sample rate)
# BACKWARD_WINDOW_LENGTH_OF_PEAK = int(0.02 / (1 / 64000))  # 20 milisec window / (1 / sample rate)
# PERCENTAGE_OF_LATE_HITS_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
# hits_dict_early = {}
# hits_dict = {}
# freqs = {}
# features_dict = {}
# surgery_index = 1
# HOP_LENGTH_DIVISION_NUMBER = 7
# N_FFT_DIVISION_NUMBER = 3.5
# COUNTER = 0
# # change sample rate 64000 adjust features windowing
# # change window size to 120 forward 20 back
# # send all matrixes to features table
# # gradient 1 and 2 of mfcc
#
# def give_fft_values(signal, sample_rate,):
#     fft_spectrum = np.fft.rfft(signal)
#     freq = np.fft.rfftfreq(signal.size, d=1. / sr)
#     fft_spectrum_abs = np.abs(fft_spectrum)
#     return fft_spectrum_abs, freq
#
# def save_periodigram_values(signal, sample_rate, class_id):
#     global freqs
#     freq, PSD = periodogram(signal, fs=sample_rate)
#     max_id = np.flip(np.argsort(PSD))[:1][0]
#     if class_id in freqs.keys():
#         freqs[class_id] = np.append(freqs[class_id], freq[max_id])
#     else:
#         freqs[class_id] = freq[max_id]
#     # sns.distplot(freqs, kde=True, label='Dominant frequency distribution of class 0')
#     # plt.show()
#
# def plot_fft(signal, str):
#     global sr
#     fft_spectrum = np.fft.rfft(signal)
#     freq = np.fft.rfftfreq(signal.size, d=1. / sr)
#     fft_spectrum_abs = np.abs(fft_spectrum)
#     plt.plot(freq, fft_spectrum_abs)
#     plt.xlabel("frequency, Hz")
#     plt.ylabel("Amplitude, units")
#     plt.title(str)
#     plt.show()
#
# def extract_features(data, sr):
#
#     global FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK, COUNTER
#     if len(data) != 0:
#         zero_cross = librosa.feature.zero_crossing_rate(
#             y=data, frame_length=int(len(data) / 3.5),
#             hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK) / HOP_LENGTH_DIVISION_NUMBER))
#         zero_cross = np.reshape(zero_cross, np.shape(zero_cross)[1])
#
#         S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
#                                                 hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK)
#                                                 / HOP_LENGTH_DIVISION_NUMBER))
#         log_s = librosa.power_to_db(S, ref=np.max)
#         log_s_values = np.reshape(log_s, np.shape(log_s)[1] * np.shape(log_s)[0])
#
#         mfcc1 = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
#                                                 hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK)
#                                                 / HOP_LENGTH_DIVISION_NUMBER))
#         mfcc_values = np.reshape(mfcc1, np.shape(mfcc1)[1] * np.shape(mfcc1)[0])
#
#         mfcc_delta = librosa.feature.delta(mfcc1, width=3, order=1, axis=0)
#         mfcc_delta_values = np.reshape(mfcc_delta, np.shape(mfcc_delta)[1] * np.shape(mfcc_delta)[0])
#
#         mfcc_delta2 = librosa.feature.delta(mfcc1, width=3, order=2, axis=0)
#         mfcc_delta2_values = np.reshape(mfcc_delta2, np.shape(mfcc_delta2)[1] * np.shape(mfcc_delta2)[0])
#
#         chromagram = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
#                                                 hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK)
#                                                 / HOP_LENGTH_DIVISION_NUMBER))
#         chromagram_values = np.reshape(chromagram, np.shape(chromagram)[1] * np.shape(chromagram)[0])
#
#         onset_env = librosa.onset.onset_strength(y=data, sr=sr)
#         tempo = librosa.beat.tempo(y=onset_env, sr=sr)
#
#         total_values = np.concatenate((zero_cross, log_s_values, mfcc_values,mfcc_delta_values, mfcc_delta2_values,
#                                        chromagram_values, tempo))
#     else:
#         total_values = None
#         COUNTER += 1
#     return total_values
#
#
# if not CSV_READ:
#     for i, file_name in enumerate(glob.glob(os.path.join(path, '*.wav'))):
#         data, sr = librosa.load(file_name, sr=64000)
#         D = np.abs(librosa.stft(data, n_fft=512))
#         onset_env = librosa.onset.onset_strength(y=data, sr=sr, aggregate=np.mean,
#                                                  S=D)
#         times = librosa.times_like(onset_env, sr=sr, hop_length=128)
#         onset_frames = librosa.onset.onset_detect(y=data, onset_envelope=onset_env, sr=sr, normalize=True)
#         # fig, ax = plt.subplots(nrows=2, sharex=True)
#         # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#         #                          x_axis='time', y_axis='log', ax=ax[0])
#         # ax[0].set(title='Power spectrogram')
#         # ax[0].label_outer()
#         # ax[1].plot(times, onset_env, label='Onset strength')
#         # ax[1].vlines(times[onset_frames], 0, onset_env.max(), color='r', alpha=0.9,
#         #              linestyle='--', label='Onsets')
#         # ax[1].legend()
#         # plt.show()
#
#
#         # onset_env = librosa.onset.onset_strength(y=data, sr=sr)
#
#         # peaks = librosa.util.peak_pick(onset_env, pre_max=300, post_max=300, pre_avg=300, post_avg=500, delta=0.5, wait=100)
#         # # times = librosa.times_like(onset_env, sr=sr, hop_length=512)
#         # fig, ax = plt.subplots(nrows=2, sharex=True)
#         # display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#         #                          y_axis='log', x_axis='time', ax=ax[1])
#         # ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
#         # ax[0].vlines(times[onset_frames], 0,
#         #              onset_env.max(), color='r', alpha=0.3,
#         #              label='Selected peaks')
#         # ax[0].legend(frameon=True, framealpha=0.8)
#         #
#         # ax[0].label_outer()
#         # plt.title(file_name)
#         # plt.show()
#         # print(i)
#         # good plot
#         # fig, ax = plt.subplots(3, 1, constrained_layout=True)
#         #
#         # librosa.display.waveshow(data, sr=sr, ax=ax[1])
#         #
#         # # fig, ax = plt.subplots(nrows=2, sharex=True)
#         # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#         #                          x_axis='time', y_axis='log', ax=ax[0])
#         # ax[0].set(title='Power spectrogram (time-frequency)')
#         # ax[0].set(xlabel='time')
#         # ax[0].label_outer()
#         #
#         # ax[2].plot(times, onset_env, label='Onset strength')
#         # ax[2].vlines(times[onset_frames], 0, onset_env.max(), color='r', alpha=0.9,
#         #              linestyle='--', label='Onsets')
#         # ax[2].legend()
#         # ax[2].set(title='Smooth signal (time)')
#         # ax[2].set(ylabel='Amplitude (abs)')
#         # ax[2].set(xlabel='time')
#         #
#         # ax[1].set(ylabel='Amplitude')
#         # ax[1].set(xlabel='time')
#         # ax[1].set(title='Raw signal (time)')
#         # plt.show()
#
#         meta_data = file_name.split('\\')[4]
#         file_id = meta_data.split('-')[0]
#         date = meta_data.split('-')[1]
#         surgon_name = meta_data.split('-')[2]
#         broach_index = meta_data.split('-')[3]
#         Relative_shaft_size = meta_data.split('-')[4].split('.')[0]
#         print(i)
#
#         total_hits_detected_in_broach = len(times[onset_frames])
#         print("total_hits_detected_in_broach:", total_hits_detected_in_broach)
#
#
#         if 'stem' in Relative_shaft_size.lower():
#             print("deleted one broach with 'stem'")
#             surgery_index += 1
#             continue
#         elif 'cerclage' in Relative_shaft_size.lower():
#             print("deleted one broach with 'cerclage'")
#             continue
#         else:
#             for hit_index, time_index in enumerate(times[onset_frames]):
#                 center_of_peak_in_time = int(time_index * sr)
#                 bottom_of_peak_in_time = int(center_of_peak_in_time - BACKWARD_WINDOW_LENGTH_OF_PEAK)
#                 top_of_peak_in_time = int(center_of_peak_in_time + FORWARD_WINDOW_LENGTH_OF_PEAK)
#                 dictionary_index = "broach_{} hit_index_{}".format(file_id, hit_index)
#                 # print("added broach_{} hit_index_{},"
#                 #       " with {} data samples".format(file_id,
#                 #                                      hit_index, data[bottom_of_peak_in_time:top_of_peak_in_time].size))
#                 # hits_dict[dictionary_index] = data[bottom_of_peak_in_time:top_of_peak_in_time]
#                 # plt.plot(hits_dict["broach_{} hit_index_{}".format(i, hit_index)])
#                 hits_dict[dictionary_index] = np.array([surgon_name, broach_index,
#                                                          Relative_shaft_size, date, surgery_index,
#                                                          total_hits_detected_in_broach])
#
#
#                 total_values = extract_features(data[bottom_of_peak_in_time:top_of_peak_in_time], sr)
#                 features_dict[dictionary_index] = total_values
#
#     print("there were ", COUNTER, " 0 data hits")
#     df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hits_dict.items()]))
#     df = df.transpose()
#     df.to_csv("DATA.csv")
#
#     df_features = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict.items()]))
#     df_features = df_features.transpose()
#     df_features.to_csv('features_data.csv')
#
#
#     # for i in freqs.keys():
#     #     sns.distplot(freqs[i], kde=True, label='Dominant frequency distribution of {}'.format(i), color='r')
#     #     plt.legend()
#     #     plt.show()
#
# else:
#
#     # df_early = pd.read_csv("DATA_early.csv")
#     df_late = pd.read_csv("DATA.csv")
#
# # df_late = df_late.rename(columns={'Unnamed: 0': 'full_id'})
#
# df_late['broach_id'] = df_late.apply(lambda row1: row1[0].split(' ')[0].split('_')[1], axis=1)
# df_late['hit_number'] = df_late.apply(lambda row1: row1[0].split(' ')[1].split('_')[-1], axis=1)
# df_late['main_index'] = df_late.apply(lambda row1: float(row1[0].split(' ')[0].split('_')[1] +
#                                                    row1[0].split(' ')[1].split('_')[-1]), axis=1)
#
# df_late['main_index'] = df_late.apply(lambda row1: row1[-1]/10 if row1[-1] > 9999 else row1[-1], axis=1)
#
# df_late = df_late.set_index(df_late.columns[0])
#
# # df_late['hit_number'].hist(bins=max(df_late['hit_number']), grid=False)
# # plt.show()
#
# df_time_data = df_late.iloc[:, :-9] # cutting meta data from df
#
#
# def extract_mfccs(data, sr):
#     # global FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK
#     # librosa.feature.zero_crossing_rate(data, int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK) / 10))
#
#     S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048)  # adjust the n_fft to lower values from 2048
#     log_s = librosa.power_to_db(S, ref=np.max)
#     log_s_values = np.mean(log_s, axis=0)
#     mfcc1 = librosa.feature.mfcc(S=log_s, n_mfcc=13)
#     mfcc_values = np.mean(mfcc1, axis=0)
#     return mfcc_values
#
#
#
# sr = 22050
#
# df_time_data["mfccs"] = df_time_data.apply(lambda x: extract_mfccs(x.to_numpy().reshape(-1, 1), sr))
# df_time_data.to_csv('mfcc_data.csv')
#
# df_time_data["features"] = df_time_data.apply(lambda x: extract_features(x.to_numpy().reshape(-1, 1), sr))
# df_time_data["features"].to_csv('features_data.csv')
#
#
# print(df_time_data["mfccs"])
#
#
# # df_to_show = pd.DataFrame(columns=df_early.columns)
# # df_late_to_show = pd.DataFrame(columns=df_late.columns)
#
#
#
# # for index, row in df_late.iterrows():
# #
# #
# #     x = row['Unnamed: 0']
# #     if "340" in row['Unnamed: 0']:
# #         df_to_show = df_to_show.append(row, ignore_index=False)
# #
# # for index, row in df_early.iterrows():
# #     x = row['Unnamed: 0']
# #     if "340" in row['Unnamed: 0']:
# #         df_to_show = df_to_show.append(row, ignore_index=False)
# #
# # print(df_to_show)
# # # plot_fft(df.iloc[0, :-4], "title")
# # fig, ax = plt.subplots(12, 1, sharey=True)
# #
# # for index, row in df_to_show.iterrows():
# #
# #     signal = df_to_show.iloc[index, 1:-4]
# #     fft_spectrum = np.fft.rfft(signal)
# #     freq = np.fft.rfftfreq(signal.size, d=1. / sr)
# #     fft_spectrum_abs = np.abs(fft_spectrum)
# #     ax[index].plot(freq, fft_spectrum_abs, label=row['Unnamed: 0'])
# #     # ax[index].xlabel("frequency, Hz")
# #     # ax[index].ylabel("Amplitude, units")
# #     # ax[index].set_title(row['Unnamed: 0'])
# #     ax[index].legend()
# # # plt.ylim()
# # plt.show()
# #
# # print("hi")
# #
# # # find difference between final hits of early broach to final hits late broach
#
#
# def update_feature_len_dict(self, log_s_values, mfcc_values, chromagram_values, beat_values):
#     features = [log_s_values, mfcc_values, chromagram_values, beat_values]
#     FEAT_SIZE = 300
#     BEAT_SIZE = 15
#     alligned_features = np.array([])
#
#     for feature in features:
#         feature_length = len(feature)
#         if feature is not beat_values:
#             if feature_length >= FEAT_SIZE:
#                 feature = np.delete(feature, [range(FEAT_SIZE, feature_length)])
#             elif feature_length < FEAT_SIZE:
#                 temp_arr = np.zeros(FEAT_SIZE - feature_length)
#                 feature = np.concatenate((feature, temp_arr), axis=0)
#         else:
#             if feature_length >= BEAT_SIZE:
#                 feature = np.delete(feature, [range(BEAT_SIZE, feature_length)])
#             elif feature_length < BEAT_SIZE:
#                 temp_arr = np.zeros(BEAT_SIZE - feature_length)
#                 feature = np.concatenate((feature, temp_arr), axis=0)
#
#         alligned_features = np.concatenate((alligned_features, feature), axis=0)
#
#     return alligned_features
#
#