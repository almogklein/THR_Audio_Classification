# from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import ShortTermFeatures
# from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
import tqdm
import glob
import os
import librosa
import time
from librosa import display
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import periodogram
import seaborn as sns

path = r"C:\Users\DELL\audiodata"
start = time.time()
CSV_READ = False
FORWARD_WINDOW_LENGTH_OF_PEAK = int(0.05 / (1 / 22050))  # 50 milisec window / (1 / sample rate)
BACKWARD_WINDOW_LENGTH_OF_PEAK = int(0.02 / (1 / 22050))  # 20 milisec window / (1 / sample rate)
PERCENTAGE_OF_LATE_HITS_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
hits_dict_early = {}
hits_dict = {}
freqs = {}
features_dict = {}
surgery_index = 1


# def give_fft_values(signal, sample_rate,):
#     fft_spectrum = np.fft.rfft(signal)
#     freq = np.fft.rfftfreq(signal.size, d=1. / sr)
#     fft_spectrum_abs = np.abs(fft_spectrum)
#     return fft_spectrum_abs, freq

def save_periodigram_values(signal, sample_rate, class_id):
    global freqs
    freq, PSD = periodogram(signal, fs=sample_rate)
    max_id = np.flip(np.argsort(PSD))[:1][0]
    if class_id in freqs.keys():
        freqs[class_id] = np.append(freqs[class_id], freq[max_id])
    else:
        freqs[class_id] = freq[max_id]
    # sns.distplot(freqs, kde=True, label='Dominant frequency distribution of class 0')
    # plt.show()

def plot_fft(signal, str):
    global sr
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / sr)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.title(str)
    plt.show()

def extract_features(data, sr):

    global FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK
    zero_cross = librosa.feature.zero_crossing_rate(data, int((FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK) / 10))
    zero_cross = np.reshape(zero_cross, np.shape(zero_cross)[1])
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048)  # adjust the n_fft to lower values from 2048
    log_s = librosa.power_to_db(S, ref=np.max)
    log_s_values = np.mean(log_s, axis=0)
    mfcc1 = librosa.feature.mfcc(S=log_s, n_mfcc=13)
    mfcc_values = np.mean(mfcc1, axis=0)
    chromagram = librosa.feature.chroma_stft(y=data, sr=sr)
    chromagram_values = np.mean(chromagram, axis=0)
    onset_env = librosa.onset.onset_strength(y=data, sr=sr)
    tempo = librosa.beat.tempo(y=onset_env, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(data)
    tempo_beats, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # aligned_features = self.update_feature_len_dict(log_s_values, mfcc_values, chromagram_values, beats)
    total_values = np.concatenate((zero_cross, log_s_values, mfcc_values, chromagram_values,
                                   beats, tempo, np.array([tempo_beats])))

    return total_values

if not CSV_READ:
    for i, file_name in enumerate(glob.glob(os.path.join(path, '*.wav'))):
        data, sr = librosa.load(file_name)
        D = np.abs(librosa.stft(data))


        onset_env = librosa.onset.onset_strength(y=data, sr=sr,
                                                 hop_length=512,
                                                 aggregate=np.median, S=D)
        times = librosa.times_like(onset_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
        #                          x_axis='time', y_axis='log', ax=ax[0])
        # ax[0].set(title='Power spectrogram')
        # ax[0].label_outer()
        # ax[1].plot(times, onset_env, label='Onset strength')
        # ax[1].vlines(times[onset_frames], 0, onset_env.max(), color='r', alpha=0.9,
        #              linestyle='--', label='Onsets')
        # ax[1].legend()
        # plt.show()


        # onset_env = librosa.onset.onset_strength(y=data, sr=sr)
        #
        # peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=100)
        # times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # display.specshow(librosa.amplitude_to_db(D, ref=np.max),
        #                          y_axis='log', x_axis='time', ax=ax[1])
        # ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
        # ax[0].vlines(times[peaks], 0,
        #              onset_env.max(), color='r', alpha=0.8,
        #              label='Selected peaks')
        # ax[0].legend(frameon=True, framealpha=0.8)
        #
        # ax[0].label_outer()
        # plt.title(file_name)
        # plt.show()
        # print(i)
        meta_data = file_name.split('\\')[4]
        file_id = meta_data.split('-')[0]
        date = meta_data.split('-')[1]
        surgon_name = meta_data.split('-')[2]
        broach_index = meta_data.split('-')[3]
        Relative_shaft_size = meta_data.split('-')[4].split('.')[0]
        print(i)
        # for PERCENTAGE_OF_LATE_HITS in PERCENTAGE_OF_LATE_HITS_LIST:
        total_hits_detected_in_broach = len(times[onset_frames])
        print("total_hits_detected_in_broach:", total_hits_detected_in_broach)
        # late_hits_number = round(total_hits_detected_in_broach * PERCENTAGE_OF_LATE_HITS)
        # late_hits_start_index = total_hits_detected_in_broach - late_hits_number
        # early_hits_end_index = late_hits_number

        if 'stem' in Relative_shaft_size.lower():
            print("deleted one broach with 'stem'")
            surgery_index += 1
            continue
        elif 'cerclage' in Relative_shaft_size.lower():
            print("deleted one broach with 'cerclage'")
            continue
        else:
            for hit_index, time_index in enumerate(times[onset_frames]):
                center_of_peak_in_time = int(time_index * sr)
                bottom_of_peak_in_time = int(center_of_peak_in_time - BACKWARD_WINDOW_LENGTH_OF_PEAK)
                top_of_peak_in_time = int(center_of_peak_in_time + FORWARD_WINDOW_LENGTH_OF_PEAK)
                dictionary_index = "broach_{} hit_index_{}".format(file_id, hit_index)
                hits_dict[dictionary_index] = data[bottom_of_peak_in_time:top_of_peak_in_time]
                # plt.plot(hits_dict["broach_{} hit_index_{}".format(i, hit_index)])
                hits_dict[dictionary_index] = np.append(hits_dict[dictionary_index],
                                                        [surgon_name, broach_index,
                                                         Relative_shaft_size, date, surgery_index,
                                                         total_hits_detected_in_broach])

                total_values = extract_features(data[bottom_of_peak_in_time:top_of_peak_in_time], sr)
                features_dict[dictionary_index] = total_values

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hits_dict.items()]))
    df = df.transpose()
    df.to_csv("DATA.csv")

    df_features = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict.items()]))
    df_features = df_features.transpose()
    df_features.to_csv('features_data.csv')
    # for i in freqs.keys():
    #     sns.distplot(freqs[i], kde=True, label='Dominant frequency distribution of {}'.format(i), color='r')
    #     plt.legend()
    #     plt.show()

else:

    # df_early = pd.read_csv("DATA_early.csv")
    df_late = pd.read_csv("DATA.csv")

# df_late = df_late.rename(columns={'Unnamed: 0': 'full_id'})

df_late['broach_id'] = df_late.apply(lambda row1: row1[0].split(' ')[0].split('_')[1], axis=1)
df_late['hit_number'] = df_late.apply(lambda row1: row1[0].split(' ')[1].split('_')[-1], axis=1)
df_late['main_index'] = df_late.apply(lambda row1: float(row1[0].split(' ')[0].split('_')[1] +
                                                   row1[0].split(' ')[1].split('_')[-1]), axis=1)

df_late['main_index'] = df_late.apply(lambda row1: row1[-1]/10 if row1[-1] > 9999 else row1[-1], axis=1)

df_late = df_late.set_index(df_late.columns[0])

# df_late['hit_number'].hist(bins=max(df_late['hit_number']), grid=False)
# plt.show()

df_time_data = df_late.iloc[:, :-9] # cutting meta data from df


def extract_mfccs(data, sr):
    # global FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK
    # librosa.feature.zero_crossing_rate(data, int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK) / 10))

    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048)  # adjust the n_fft to lower values from 2048
    log_s = librosa.power_to_db(S, ref=np.max)
    log_s_values = np.mean(log_s, axis=0)
    mfcc1 = librosa.feature.mfcc(S=log_s, n_mfcc=13)
    mfcc_values = np.mean(mfcc1, axis=0)
    return mfcc_values



sr = 22050

df_time_data["mfccs"] = df_time_data.apply(lambda x: extract_mfccs(x.to_numpy().reshape(-1, 1), sr))
df_time_data.to_csv('mfcc_data.csv')

df_time_data["features"] = df_time_data.apply(lambda x: extract_features(x.to_numpy().reshape(-1, 1), sr))
df_time_data["features"].to_csv('features_data.csv')


print(df_time_data["mfccs"])


# df_to_show = pd.DataFrame(columns=df_early.columns)
# df_late_to_show = pd.DataFrame(columns=df_late.columns)



# for index, row in df_late.iterrows():
#
#
#     x = row['Unnamed: 0']
#     if "340" in row['Unnamed: 0']:
#         df_to_show = df_to_show.append(row, ignore_index=False)
#
# for index, row in df_early.iterrows():
#     x = row['Unnamed: 0']
#     if "340" in row['Unnamed: 0']:
#         df_to_show = df_to_show.append(row, ignore_index=False)
#
# print(df_to_show)
# # plot_fft(df.iloc[0, :-4], "title")
# fig, ax = plt.subplots(12, 1, sharey=True)
#
# for index, row in df_to_show.iterrows():
#
#     signal = df_to_show.iloc[index, 1:-4]
#     fft_spectrum = np.fft.rfft(signal)
#     freq = np.fft.rfftfreq(signal.size, d=1. / sr)
#     fft_spectrum_abs = np.abs(fft_spectrum)
#     ax[index].plot(freq, fft_spectrum_abs, label=row['Unnamed: 0'])
#     # ax[index].xlabel("frequency, Hz")
#     # ax[index].ylabel("Amplitude, units")
#     # ax[index].set_title(row['Unnamed: 0'])
#     ax[index].legend()
# # plt.ylim()
# plt.show()
#
# print("hi")
#
# # find difference between final hits of early broach to final hits late broach


def update_feature_len_dict(self, log_s_values, mfcc_values, chromagram_values, beat_values):
    features = [log_s_values, mfcc_values, chromagram_values, beat_values]
    FEAT_SIZE = 300
    BEAT_SIZE = 15
    alligned_features = np.array([])

    for feature in features:
        feature_length = len(feature)
        if feature is not beat_values:
            if feature_length >= FEAT_SIZE:
                feature = np.delete(feature, [range(FEAT_SIZE, feature_length)])
            elif feature_length < FEAT_SIZE:
                temp_arr = np.zeros(FEAT_SIZE - feature_length)
                feature = np.concatenate((feature, temp_arr), axis=0)
        else:
            if feature_length >= BEAT_SIZE:
                feature = np.delete(feature, [range(BEAT_SIZE, feature_length)])
            elif feature_length < BEAT_SIZE:
                temp_arr = np.zeros(BEAT_SIZE - feature_length)
                feature = np.concatenate((feature, temp_arr), axis=0)

        alligned_features = np.concatenate((alligned_features, feature), axis=0)

    return alligned_features
