from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb
import json
from wfdb import processing, io
import pywt
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import stft
from utils import *

"""generate the effective record file list and dictionary for ST episodes"""
atr_list = generate_atr_list('./ltst_records/', False)
# _, ltst_dict = create_st_dictionary('./ltst_records', False, 'stb')
# with open('ltst_st_episodes.json', 'w') as outfile:
#     json.dump(ltst_dict, outfile)

"""load the ST episode dictionary"""
with open('./ltst_st_episodes.json') as infile:
    ltst_dict = json.load(infile)

point_mode = 'partial'  # 'partial' or 'all'
c_st = 0
c_n = 0

# for atr in atr_list:
#     sig_st_dict = load_st_signal(atr)
#     sig_n_dict = get_normal_part(atr)
#     sig_dict = dict(sig_st_dict, **sig_n_dict)
#     print("++++++Current record {}++++++".format(atr))
#     print(sig_dict)

sig_st_dict = load_st_signal('s20011', ltst_dict, './ltst_records/', False)
sig_n_dict = get_normal_part('s20011', ltst_dict, './ltst_records/')
sig_dict = dict(sig_st_dict, **sig_n_dict)
print(sig_dict)

for key in sig_dict.keys():
    if sig_dict[key]:
        print("+++Current Type {} and Current length {}+++".format(key, len(sig_dict[key][0])))

        sig = sig_dict[key]

        # Removal of baseline wandering using wavelet
        DenoisingSignal = remove_baseline(sig, 'bior2.6', False)

        try:

            """
            Binary spline wavelet filtering, level: 4,
            low-pass filter：[1/4 3/4 3/4 1/4], high-pass filter：[-1/4 -3/4 3/4 1/4]
            """
            points = 1000 if point_mode == 'partial' else DenoisingSignal.shape[0]
            level = 4
            ecgdata = DenoisingSignal
            swa = np.zeros((4, points), dtype=np.float32)
            swd = np.zeros((4, points), dtype=np.float32)
            signal = ecgdata[0:points]
            swa, swd = binary_spline_wavelet_filter(swa, swd, signal, points, 4, False)

            """calculate the positive and negative maximal value pairs"""
            w_peak = get_maximal_value_pairs(swd, points, level, False)

            interval2 = np.zeros_like(signal)
            intervalqs = np.zeros_like(signal)
            Mj1 = w_peak[0, :]
            Mj3 = w_peak[2, :]
            Mj4 = w_peak[3, :]

            """determine if it is inverted, use Mj4"""
            _, _, diff_inv = get_difference(w_peak, points, 4)
            print(np.where(np.asarray(diff_inv) == 2)[0].shape[0], np.nonzero(np.asarray(diff_inv))[0].shape[0])
            if np.where(np.asarray(diff_inv) == 2)[0].shape[0] > 0.5 * np.nonzero(np.asarray(diff_inv))[0].shape[0]:
                print("Inverted signal! Skip it!")
                continue

            interval, loca, diff = get_difference(w_peak, points, 3)
            loca2 = np.where(np.array(diff) == -2)[0]
            interval2[loca[loca2]] = interval[loca[loca2]]
            interval2[loca[loca2 + 1]] = interval[loca[loca2 + 1]]
            intervalqs[0:points - 10] = interval2[10:points]

            """Q, R, S, onset, offset detection for each QRS complex"""
            Q_located, R_located, S_located, onset_located, offset_located = qrs_delineation(w_peak, points, interval2)

            R_R_interval = int(np.mean(R_located[1:] - R_located[:-1]))
            print("      mean distance between two successive R peaks:", R_R_interval)

            """find T peaks and pop error points"""
            T_located, pop_list_t = find_t_peak(signal, R_R_interval, offset_located)
            R_located = np.delete(R_located, pop_list_t)
            S_located = np.delete(S_located, pop_list_t)
            Q_located = np.delete(Q_located, pop_list_t)
            onset_located = np.delete(onset_located, pop_list_t)
            offset_located = np.delete(offset_located, pop_list_t)

            """find P peaks and pop error points"""
            P_located, pop_list_p = find_p_peak(signal, R_R_interval, onset_located)
            R_located = np.delete(R_located, pop_list_p)
            S_located = np.delete(S_located, pop_list_p)
            Q_located = np.delete(Q_located, pop_list_p)
            onset_located = np.delete(onset_located, pop_list_p)
            offset_located = np.delete(offset_located, pop_list_p)
            T_located = np.delete(T_located, pop_list_p)

            # calculate the reference voltage to find F points
            mean_idx_diff_1, mean_idx_diff_2, reference_voltage = get_reference_voltage(signal, R_located,
                                                                                        onset_located,
                                                                                        offset_located)
            """find F points and and pop error points"""
            F_located, F_located_val, pop_list_f = find_f_point(signal, offset_located, T_located, reference_voltage)
            if np.where(F_located_val == 1)[0].shape[0] == F_located.shape[0]:
                print("All abnormal F points! Drop it!")
                continue

            R_located = np.delete(R_located, pop_list_f)
            S_located = np.delete(S_located, pop_list_f)
            Q_located = np.delete(Q_located, pop_list_f)
            onset_located = np.delete(onset_located, pop_list_f)
            offset_located = np.delete(offset_located, pop_list_f)
            T_located = np.delete(T_located, pop_list_f)
            P_located = np.delete(P_located, pop_list_f)

            """Plot the delineation"""
            print(R_located.shape, S_located.shape, Q_located.shape, onset_located.shape, offset_located.shape,
                  T_located.shape, P_located.shape, F_located.shape)
            # plt.figure(6)
            # plt.plot(signal)
            # plt.plot(R_located, signal[R_located], 'rx', marker='x', color='r', label='R', markersize=8)
            # plt.plot(onset_located, signal[onset_located], 'rx', marker='o', color='k', label='Onset', markersize=6)
            # plt.plot(offset_located, signal[offset_located], 'rx', marker='o', color='k', label='Offset', markersize=6)
            # plt.plot(P_located, signal[P_located], 'rx', marker='v', color='g', label='P', markersize=6)
            # plt.plot(T_located, signal[T_located], 'rx', marker='v', color='g', label='T', markersize=6)
            # plt.plot(F_located, signal[F_located], 'rx', marker='^', color='c', label='F', markersize=8)
            # plt.xlabel('Sample')
            # plt.ylabel('Voltage V')
            # plt.title('QRSPT Delineation')
            # plt.legend(loc='best')
            # plt.show()

            """features from paper Ischemia episode detection......"""
            x_1, x_2, x_3 = generate_features(signal, R_located, T_located, F_located, F_located_val,
                                              mean_idx_diff_1, mean_idx_diff_2, reference_voltage)

            """features from paper ECG features and methods......"""
            """morphological features based on QRSPT delineation"""
            qrs_i = S_located - Q_located
            qt_i = T_located - Q_located
            pq_i = Q_located - P_located
            jt_i = T_located - offset_located
            qrs_v = signal[R_located]
            t_v = signal[T_located]
            st20_v = signal[offset_located + 5]  # 20ms * 1000 / 250 = 5

            """morphological features without QRSPT delineation"""
            qrs_argmax = []
            qrs_argmin = []
            for i in range(R_located.shape[0]):
                qrs_argmax.append(np.argmax(signal[R_located[i] - int(R_R_interval / 6):
                                                   R_located[i] + int(R_R_interval / 6 * 5)]) +
                                  R_located[i] - int(R_R_interval / 6))
                qrs_argmin.append(np.argmin(signal[R_located[i] - int(R_R_interval / 6):
                                                   R_located[i] + int(R_R_interval / 6 * 5)]) +
                                  R_located[i] - int(R_R_interval / 6))
            qrs_argmax = np.asarray(qrs_argmax).reshape([len(qrs_argmax), 1])
            qrs_argmin = np.asarray(qrs_argmin).reshape([len(qrs_argmin), 1])

            qrs_max = signal[qrs_argmax]
            qrs_min = signal[qrs_argmin]
            qrs_max_min = qrs_max - qrs_min
            qrs_argmax_argmin = qrs_argmax - qrs_argmin

            qrs_a = []
            qrst_a = []
            pqrst_a = []
            for i in range(R_located.shape[0]):
                qrs_a.append(np.sum(signal[R_located[i] - int(R_R_interval / 12):
                                           R_located[i] + int(R_R_interval / 12)]) * int(R_R_interval / 6))
                qrst_a.append(np.sum(signal[R_located[i] - int(R_R_interval / 12):
                                            R_located[i] + int(R_R_interval / 4)]) * int(R_R_interval / 3))
                pqrst_a.append(np.sum(signal[R_located[i] - int(R_R_interval / 6):
                                             R_located[i] + int(R_R_interval / 6 * 5)]) * R_R_interval)
            qrs_a = np.asarray(qrs_a).reshape([len(qrs_a), 1])
            qrst_a = np.asarray(qrst_a).reshape([len(qrst_a), 1])
            pqrst_a = np.asarray(pqrst_a).reshape([len(pqrst_a), 1])

            qrs_qrst_ratio = qrs_a / qrst_a
            qrs_pqrst_ratio = qrs_a / pqrst_a
            qrst_pqrst_ratio = qrst_a / pqrst_a

            """spectral features without QRSPT delineation"""
            fft_f1 = []
            fft_f2 = []
            fft_f3 = []
            stft_mean = []
            stft_median = []
            stft_max = []
            for i in range(R_located.shape[0]):
                """FFT"""
                each_beat = signal[R_located[i] - int(R_R_interval / 6):
                                   R_located[i] + int(R_R_interval / 6 * 5) + 1]
                fftv = rfft(each_beat)
                freq = rfftfreq(R_R_interval, 1. / 500)
                mask = freq > 0
                fft_theo = 2.0 * np.abs(fftv / R_R_interval)
                # plt.figure(i)
                # ax1 = plt.subplot2grid((2, 1), (0, 0))
                # ax1.plot(each_beat, 'r')
                # ax1.set_title("Original signal in time domain")
                # ax2 = plt.subplot2grid((2, 1), (1, 0))
                # ax2.plot(freq[mask], fft_theo[mask], 'g')
                # ax2.set_title("Spectrum")
                # plt.show()
                fft_f1.append(np.sum(fft_theo[:35]))
                fft_f2.append(np.sum(fft_theo[35:90]))
                fft_f3.append(np.sum(fft_theo[125:250]))

                """STFT"""
                f, t, Zxx = stft(each_beat, 500, nperseg=R_R_interval)
                Zxx = np.abs(Zxx)
                stft_mean.append(np.mean(Zxx))
                stft_median.append(np.median(Zxx))
                stft_max.append(np.max(Zxx))
                # plt.pcolormesh(t, f, np.abs(Zxx))
                # plt.title('STFT Magnitude')
                # plt.ylabel('Frequency [Hz]')
                # plt.xlabel('Time [sec]')
                # plt.show()
            fft_f1 = np.asarray(fft_f1).reshape([len(fft_f1), 1])
            fft_f2 = np.asarray(fft_f2).reshape([len(fft_f2), 1])
            fft_f3 = np.asarray(fft_f3).reshape([len(fft_f3), 1])
            stft_mean = np.asarray(stft_mean).reshape([len(stft_mean), 1])
            stft_median = np.asarray(stft_median).reshape([len(stft_median), 1])
            stft_max = np.asarray(stft_max).reshape([len(stft_max), 1])

            """divide by 5"""
            qrs_il = []
            qt_il = []
            pq_il = []
            jt_il = []
            qrs_vl = []
            t_vl = []
            st20_vl = []
            qrs_argmaxl = []
            qrs_argminl = []
            qrs_maxl = []
            qrs_minl = []
            qrs_max_minl = []
            qrs_argmax_argminl = []
            qrs_al = []
            qrst_al = []
            pqrst_al = []
            qrs_qrst_ratiol = []
            qrs_pqrst_ratiol = []
            qrst_pqrst_ratiol = []
            fft_f1l = []
            fft_f2l = []
            fft_f3l = []
            stft_meanl = []
            stft_medianl = []
            stft_maxl = []
            nbeat = R_located.shape[0]
            for i in range(int(nbeat / 5)):
                qrs_il.append(np.sum(qrs_i[5 * i:5 * i + 5]) / 5)
                qt_il.append(np.sum(qt_i[5 * i:5 * i + 5]) / 5)
                pq_il.append(np.sum(pq_i[5 * i:5 * i + 5]) / 5)
                jt_il.append(np.sum(jt_i[5 * i:5 * i + 5]) / 5)
                qrs_vl.append(np.sum(qrs_v[5 * i:5 * i + 5]) / 5)
                t_vl.append(np.sum(t_v[5 * i:5 * i + 5]) / 5)
                st20_vl.append(np.sum(st20_v[5 * i:5 * i + 5]) / 5)
                qrs_argmaxl.append(np.sum(qrs_argmax[5 * i:5 * i + 5]) / 5)
                qrs_argminl.append(np.sum(qrs_argmin[5 * i:5 * i + 5]) / 5)
                qrs_maxl.append(np.sum(qrs_max[5 * i:5 * i + 5]) / 5)
                qrs_minl.append(np.sum(qrs_min[5 * i:5 * i + 5]) / 5)
                qrs_max_minl.append(np.sum(qrs_max_min[5 * i:5 * i + 5]) / 5)
                qrs_argmax_argminl.append(np.sum(qrs_argmax_argmin[5 * i:5 * i + 5]) / 5)
                qrs_al.append(np.sum(qrs_a[5 * i:5 * i + 5]) / 5)
                qrst_al.append(np.sum(qrst_a[5 * i:5 * i + 5]) / 5)
                pqrst_al.append(np.sum(pqrst_a[5 * i:5 * i + 5]) / 5)
                qrs_qrst_ratiol.append(np.sum(qrs_qrst_ratio[5 * i:5 * i + 5]) / 5)
                qrs_pqrst_ratiol.append(np.sum(qrs_pqrst_ratio[5 * i:5 * i + 5]) / 5)
                qrst_pqrst_ratiol.append(np.sum(qrst_pqrst_ratio[5 * i:5 * i + 5]) / 5)
                fft_f1l.append(np.sum(fft_f1[5 * i:5 * i + 5]) / 5)
                fft_f2l.append(np.sum(fft_f2[5 * i:5 * i + 5]) / 5)
                fft_f3l.append(np.sum(fft_f3[5 * i:5 * i + 5]) / 5)
                stft_meanl.append(np.sum(stft_mean[5 * i:5 * i + 5]) / 5)
                stft_medianl.append(np.sum(stft_median[5 * i:5 * i + 5]) / 5)
                stft_maxl.append(np.sum(stft_max[5 * i:5 * i + 5]) / 5)

            if nbeat % 5 != 0:
                qrs_il.append(np.sum(qrs_i[int(nbeat / 5) * 5 - len(qrs_i):]) /
                              (len(qrs_i) - int(nbeat / 5) * 5))
                qt_il.append(np.sum(qt_i[int(nbeat / 5) * 5 - len(qt_i):]) /
                             (len(qt_i) - int(nbeat / 5) * 5))
                pq_il.append(np.sum(pq_i[int(nbeat / 5) * 5 - len(pq_i):]) /
                             (len(pq_i) - int(nbeat / 5) * 5))
                jt_il.append(np.sum(jt_i[int(nbeat / 5) * 5 - len(jt_i):]) /
                             (len(jt_i) - int(nbeat / 5) * 5))
                qrs_vl.append(np.sum(qrs_v[int(nbeat / 5) * 5 - len(qrs_v):]) /
                              (len(qrs_v) - int(nbeat / 5) * 5))
                t_vl.append(np.sum(t_v[int(nbeat / 5) * 5 - len(t_v):]) /
                            (len(t_v) - int(nbeat / 5) * 5))
                st20_vl.append(np.sum(st20_v[int(nbeat / 5) * 5 - len(st20_v):]) /
                               (len(st20_v) - int(nbeat / 5) * 5))
                qrs_argmaxl.append(np.sum(qrs_argmax[int(nbeat / 5) * 5 - len(qrs_argmax):]) /
                                   (len(qrs_argmax) - int(nbeat / 5) * 5))
                qrs_argminl.append(np.sum(qrs_argmin[int(nbeat / 5) * 5 - len(qrs_argmin):]) /
                                   (len(qrs_argmin) - int(nbeat / 5) * 5))
                qrs_maxl.append(np.sum(qrs_max[int(nbeat / 5) * 5 - len(qrs_max):]) /
                                (len(qrs_max) - int(nbeat / 5) * 5))
                qrs_minl.append(np.sum(qrs_min[int(nbeat / 5) * 5 - len(qrs_min):]) /
                                (len(qrs_min) - int(nbeat / 5) * 5))
                qrs_max_minl.append(np.sum(qrs_max_min[int(nbeat / 5) * 5 - len(qrs_max_min):]) /
                                    (len(qrs_max_min) - int(nbeat / 5) * 5))
                qrs_argmax_argminl.append(np.sum(qrs_argmax_argmin[int(nbeat / 5) * 5 - len(qrs_argmax_argmin):]) /
                                          (len(qrs_argmax_argmin) - int(nbeat / 5) * 5))
                qrs_al.append(np.sum(qrs_a[int(nbeat / 5) * 5 - len(qrs_a):]) /
                              (len(qrs_a) - int(nbeat / 5) * 5))
                qrst_al.append(np.sum(qrst_a[int(nbeat / 5) * 5 - len(qrst_a):]) /
                               (len(qrst_a) - int(nbeat / 5) * 5))
                pqrst_al.append(np.sum(pqrst_a[int(nbeat / 5) * 5 - len(pqrst_a):]) /
                                (len(pqrst_a) - int(nbeat / 5) * 5))
                qrs_qrst_ratiol.append(np.sum(qrs_qrst_ratio[int(nbeat / 5) * 5 - len(qrs_qrst_ratio):]) /
                                       (len(qrs_qrst_ratio) - int(nbeat / 5) * 5))
                qrs_pqrst_ratiol.append(np.sum(qrs_pqrst_ratio[int(nbeat / 5) * 5 - len(qrs_pqrst_ratio):]) /
                                        (len(qrs_pqrst_ratio) - int(nbeat / 5) * 5))
                qrst_pqrst_ratiol.append(np.sum(qrst_pqrst_ratio[int(nbeat / 5) * 5 - len(qrst_pqrst_ratio):]) /
                                         (len(qrst_pqrst_ratio) - int(nbeat / 5) * 5))
                fft_f1l.append(np.sum(fft_f1[int(nbeat / 5) * 5 - len(fft_f1):]) /
                               (len(fft_f1) - int(nbeat / 5) * 5))
                fft_f2l.append(np.sum(fft_f2[int(nbeat / 5) * 5 - len(fft_f2):]) /
                               (len(fft_f2) - int(nbeat / 5) * 5))
                fft_f3l.append(np.sum(fft_f3[int(nbeat / 5) * 5 - len(fft_f3):]) /
                               (len(fft_f3) - int(nbeat / 5) * 5))
                stft_meanl.append(np.sum(stft_mean[int(nbeat / 5) * 5 - len(stft_mean):]) /
                                  (len(stft_mean) - int(nbeat / 5) * 5))
                stft_medianl.append(np.sum(stft_median[int(nbeat / 5) * 5 - len(stft_median):]) /
                                    (len(stft_median) - int(nbeat / 5) * 5))
                stft_maxl.append(np.sum(stft_max[int(nbeat / 5) * 5 - len(stft_max):]) /
                                (len(stft_max) - int(nbeat / 5) * 5))

            qrs_il = np.asarray(qrs_il).reshape([len(qrs_il), 1])
            qt_il = np.asarray(qt_il).reshape([len(qt_il), 1])
            pq_il = np.asarray(pq_il).reshape([len(pq_il), 1])
            jt_il = np.asarray(jt_il).reshape([len(jt_il), 1])
            qrs_vl = np.asarray(qrs_vl).reshape([len(qrs_vl), 1])
            t_vl = np.asarray(t_vl).reshape([len(t_vl), 1])
            st20_vl = np.asarray(st20_vl).reshape([len(st20_vl), 1])
            qrs_argmaxl = np.asarray(qrs_argmaxl).reshape([len(qrs_argmaxl), 1])
            qrs_argminl = np.asarray(qrs_argminl).reshape([len(qrs_argminl), 1])
            qrs_maxl = np.asarray(qrs_maxl).reshape([len(qrs_maxl), 1])
            qrs_minl = np.asarray(qrs_minl).reshape([len(qrs_minl), 1])
            qrs_max_minl = np.asarray(qrs_max_minl).reshape([len(qrs_max_minl), 1])
            qrs_argmax_argminl = np.asarray(qrs_argmax_argminl).reshape([len(qrs_argmax_argminl), 1])
            qrs_al = np.asarray(qrs_al).reshape([len(qrs_al), 1])
            qrst_al = np.asarray(qrst_al).reshape([len(qrst_al), 1])
            pqrst_al = np.asarray(pqrst_al).reshape([len(pqrst_al), 1])
            qrs_qrst_ratiol = np.asarray(qrs_qrst_ratiol).reshape([len(qrs_qrst_ratiol), 1])
            qrs_pqrst_ratiol = np.asarray(qrs_pqrst_ratiol).reshape([len(qrs_pqrst_ratiol), 1])
            qrst_pqrst_ratiol = np.asarray(qrst_pqrst_ratiol).reshape([len(qrst_pqrst_ratiol), 1])
            fft_f1l = np.asarray(fft_f1l).reshape([len(fft_f1l), 1])
            fft_f2l = np.asarray(fft_f2l).reshape([len(fft_f2l), 1])
            fft_f3l = np.asarray(fft_f3l).reshape([len(fft_f3l), 1])
            stft_meanl = np.asarray(stft_meanl).reshape([len(stft_meanl), 1])
            stft_medianl = np.asarray(stft_medianl).reshape([len(stft_medianl), 1])
            stft_maxl = np.asarray(stft_maxl).reshape([len(stft_maxl), 1])

            label = np.ones([x_1.shape[0], 1]) if key in ['ST0', 'ST1', 'ST2'] else np.zeros([x_1.shape[0], 1])
            a = np.concatenate((x_1, x_2, x_3, qrs_il, qt_il, pq_il, jt_il, qrs_vl, t_vl, st20_vl, qrs_argmaxl,
                                qrs_argminl, qrs_maxl, qrs_minl, qrs_max_minl, qrs_argmax_argminl, qrs_al,
                                qrst_al, pqrst_al, qrs_qrst_ratiol, qrs_pqrst_ratiol, qrst_pqrst_ratiol,
                                fft_f1l, fft_f2l, fft_f3l, stft_meanl, stft_medianl, stft_maxl, label), axis=1)

            if key in ['ST0', 'ST1', 'ST2']:
                c_st += a.shape[0]
            else:
                c_n += a.shape[0]

            print("      Generated numpy array has shape:", a.shape)

            np.save('./trainingsets/{}_{}.npy'.format('s20011', key), a)

        except:
            print("---Record {}, Type {} failed---".format('s20011', key))

print("All finished! Total ST samples {} and normal samples {}".format(c_st, c_n))