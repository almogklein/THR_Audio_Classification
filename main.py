import numpy as np
import pandas as pd
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)


# # find difference between final hits of early broach to final hits late bro
def load_metadata_database(path, CSV_READ, PLOT, sample_rate, HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER):

    FORWARD_WINDOW_LENGTH_OF_PEAK = int(0.12 / (1 / sample_rate))  # 120 milisec window / (1 / sample rate)
    BACKWARD_WINDOW_LENGTH_OF_PEAK = int(0.02 / (1 / sample_rate))  # 20 milisec window / (1 / sample rate)
    NO_BACKWARD_WINDOW = 0
    COUNTER = 0
    surgery_index = 1
    hits_dict = {}
    features_dict_Backward_Forward = {}
    features_dict_Forward = {}

    if CSV_READ:
        for i, file_name in enumerate(glob.glob(os.path.join(path, '*.wav'))):

            data, sr = librosa.load(file_name, sr=sample_rate)
            D = np.abs(librosa.stft(data, n_fft=2048, hop_length=1024, window="hann"))
            onset_env = librosa.onset.onset_strength(y=data, sr=sr, aggregate=np.mean,
                                                     S=D)
            times = librosa.times_like(onset_env, sr=sr, n_fft=2048, hop_length=1024)
            onset_frames = librosa.onset.onset_detect(y=data, onset_envelope=onset_env, sr=sr, normalize=True)

            if PLOT:
                plot_graph_time_freq_peak(D, data, sr, times, onset_env, onset_frames)

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

                    total_values_Backward_Forward = extract_features(data[bottom_of_peak_in_time:top_of_peak_in_time], sr,
                                                    FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK,
                                                    HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER, COUNTER)

                    total_values_Forward = extract_features(data[center_of_peak_in_time:top_of_peak_in_time], sr,
                                                       FORWARD_WINDOW_LENGTH_OF_PEAK, NO_BACKWARD_WINDOW,
                                                       HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER, COUNTER)

                    features_dict_Backward_Forward[dictionary_index] = total_values_Backward_Forward
                    features_dict_Forward[dictionary_index] = total_values_Forward

        print("there were ", COUNTER, " 0 data hits")

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hits_dict.items()]))
        df = df.transpose()
        df.to_csv("DATA.csv")

        df_features_Backward_Forward = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict_Backward_Forward.items()]))
        df_features_Backward_Forward = df_features_Backward_Forward.transpose()
        df_features_Backward_Forward.to_csv('features_data_Backward_Forward.csv')

        df_features_Forward = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in features_dict_Forward.items()]))
        df_features_Forward = df_features_Forward.transpose()
        df_features_Forward.to_csv('features_data_Forward.csv')


def extract_features(data, sr, FORWARD_WINDOW_LENGTH_OF_PEAK, BACKWARD_WINDOW_LENGTH_OF_PEAK,
                     HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER, COUNTER):

    if len(data) != 0:
        zero_cross = librosa.feature.zero_crossing_rate(
            y=data, frame_length=int(len(data) / N_FFT_DIVISION_NUMBER),
            hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK) / HOP_LENGTH_DIVISION_NUMBER))
        zero_cross = np.reshape(zero_cross, np.shape(zero_cross)[1])

        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                                hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK+BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                / HOP_LENGTH_DIVISION_NUMBER))
        log_s = librosa.power_to_db(S, ref=np.max)
        log_s_values = np.reshape(log_s, np.shape(log_s)[1] * np.shape(log_s)[0])

        mfcc1 = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                                hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                / HOP_LENGTH_DIVISION_NUMBER))
        mfcc_values = np.reshape(mfcc1, np.shape(mfcc1)[1] * np.shape(mfcc1)[0])

        mfcc_delta = librosa.feature.delta(mfcc1, width=3, order=1, axis=0)
        mfcc_delta_values = np.reshape(mfcc_delta, np.shape(mfcc_delta)[1] * np.shape(mfcc_delta)[0])

        mfcc_delta2 = librosa.feature.delta(mfcc1, width=3, order=2, axis=0)
        mfcc_delta2_values = np.reshape(mfcc_delta2, np.shape(mfcc_delta2)[1] * np.shape(mfcc_delta2)[0])

        chromagram = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=int(len(data) / N_FFT_DIVISION_NUMBER),
                                                hop_length=int((FORWARD_WINDOW_LENGTH_OF_PEAK + BACKWARD_WINDOW_LENGTH_OF_PEAK)
                                                / HOP_LENGTH_DIVISION_NUMBER))
        chromagram_values = np.reshape(chromagram, np.shape(chromagram)[1] * np.shape(chromagram)[0])

        onset_env = librosa.onset.onset_strength(y=data, sr=sr)
        tempo = librosa.beat.tempo(y=onset_env, sr=sr)

        total_values = np.concatenate((zero_cross, log_s_values, mfcc_values, mfcc_delta_values, mfcc_delta2_values,
                                       chromagram_values, tempo))
    else:
        total_values = None
        COUNTER += 1
    return total_values


def plot_graph_time_freq_peak(D, data, sr, times, onset_env, onset_frames):

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


if __name__ == '__main__':

    path = r"C:\Users\almog\data_thr\raw data"
    CSV_READ = True
    PLOT = False
    sample_rate = 64000
    HOP_LENGTH_DIVISION_NUMBER = 7
    N_FFT_DIVISION_NUMBER = 3.5

    load_metadata_database(path, CSV_READ, PLOT, sample_rate, HOP_LENGTH_DIVISION_NUMBER, N_FFT_DIVISION_NUMBER)
