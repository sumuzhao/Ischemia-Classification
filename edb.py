# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import stft
import wfdb
import json
from wfdb.processing import find_local_peaks
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import pywt
from utils import *

atr_list = generate_atr_list('./edb_records')
with open('./edb_st_episodes.json') as infile:
    st_dict = json.load(infile)

point_mode = 'all'  # 'partial' or 'all'
# key = 'N1'  # 'ST0-', 'ST1-', 'N0', 'N1'
c_st = 0
c_n = 0

train_dict = {}
for atr in atr_list:
    sig_st_dict = load_st_signal(atr, st_dict)
    sig_n_dict = get_normal_part(atr, st_dict)
    sig_dict = dict(sig_st_dict, **sig_n_dict)
    print("++++++Current record {}++++++".format(atr))
    print(sig_dict)

# sig_st_dict = load_st_signal('e0606', st_dict)
# sig_n_dict = get_normal_part('e0606', st_dict)
# sig_dict = dict(sig_st_dict, **sig_n_dict)

    train_dict[atr] = {}

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
                points = 10000 if point_mode == 'partial' else DenoisingSignal.shape[0]
                level = 4
                swa = np.zeros((4, points), dtype=np.float32)
                swd = np.zeros((4, points), dtype=np.float32)
                signal = DenoisingSignal[0:points]
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
                plt.figure(6)
                plt.plot(signal)
                plt.plot(R_located, signal[R_located], 'rx', marker='x', color='r', label='R', markersize=8)
                plt.plot(onset_located, signal[onset_located], 'rx', marker='o', color='k', label='Onset', markersize=6)
                plt.plot(offset_located, signal[offset_located], 'rx', marker='o', color='k', label='Offset', markersize=6)
                plt.plot(P_located, signal[P_located], 'rx', marker='v', color='g', label='P', markersize=6)
                plt.plot(T_located, signal[T_located], 'rx', marker='v', color='g', label='T', markersize=6)
                plt.plot(F_located, signal[F_located], 'rx', marker='^', color='c', label='F', markersize=8)

                plt.xlabel('Sample')
                plt.ylabel('Voltage V')
                plt.legend(loc='best')
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
                if offset_located[-1] + 5 < signal.shape[0]:
                    st20_v = signal[offset_located + 5]  # 20ms * 1000 / 250 = 5
                else:
                    st20_v = signal[offset_located[:-1] + 5]
                    st20_v = np.concatenate((st20_v, signal[-1].reshape([1, ])), axis=0)

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

                """averaging, divide by 5"""
                # qrs_il = []
                # qt_il = []
                # pq_il = []
                # jt_il = []
                # qrs_vl = []
                # t_vl = []
                # st20_vl = []
                # qrs_argmaxl = []
                # qrs_argminl = []
                # qrs_maxl = []
                # qrs_minl = []
                # qrs_max_minl = []
                # qrs_argmax_argminl = []
                # qrs_al = []
                # qrst_al = []
                # pqrst_al = []
                # qrs_qrst_ratiol = []
                # qrs_pqrst_ratiol = []
                # qrst_pqrst_ratiol = []
                # fft_f1l = []
                # fft_f2l = []
                # fft_f3l = []
                # stft_meanl = []
                # stft_medianl = []
                # stft_maxl = []
                # nbeat = R_located.shape[0]
                #
                # for i in range(int(nbeat / 5)):
                #     qrs_il.append(np.sum(qrs_i[5 * i:5 * i + 5]) / 5)
                #     qt_il.append(np.sum(qt_i[5 * i:5 * i + 5]) / 5)
                #     pq_il.append(np.sum(pq_i[5 * i:5 * i + 5]) / 5)
                #     jt_il.append(np.sum(jt_i[5 * i:5 * i + 5]) / 5)
                #     qrs_vl.append(np.sum(qrs_v[5 * i:5 * i + 5]) / 5)
                #     t_vl.append(np.sum(t_v[5 * i:5 * i + 5]) / 5)
                #     st20_vl.append(np.sum(st20_v[5 * i:5 * i + 5]) / 5)
                #     qrs_argmaxl.append(np.sum(qrs_argmax[5 * i:5 * i + 5]) / 5)
                #     qrs_argminl.append(np.sum(qrs_argmin[5 * i:5 * i + 5]) / 5)
                #     qrs_maxl.append(np.sum(qrs_max[5 * i:5 * i + 5]) / 5)
                #     qrs_minl.append(np.sum(qrs_min[5 * i:5 * i + 5]) / 5)
                #     qrs_max_minl.append(np.sum(qrs_max_min[5 * i:5 * i + 5]) / 5)
                #     qrs_argmax_argminl.append(np.sum(qrs_argmax_argmin[5 * i:5 * i + 5]) / 5)
                #     qrs_al.append(np.sum(qrs_a[5 * i:5 * i + 5]) / 5)
                #     qrst_al.append(np.sum(qrst_a[5 * i:5 * i + 5]) / 5)
                #     pqrst_al.append(np.sum(pqrst_a[5 * i:5 * i + 5]) / 5)
                #     qrs_qrst_ratiol.append(np.sum(qrs_qrst_ratio[5 * i:5 * i + 5]) / 5)
                #     qrs_pqrst_ratiol.append(np.sum(qrs_pqrst_ratio[5 * i:5 * i + 5]) / 5)
                #     qrst_pqrst_ratiol.append(np.sum(qrst_pqrst_ratio[5 * i:5 * i + 5]) / 5)
                #     fft_f1l.append(np.sum(fft_f1[5 * i:5 * i + 5]) / 5)
                #     fft_f2l.append(np.sum(fft_f2[5 * i:5 * i + 5]) / 5)
                #     fft_f3l.append(np.sum(fft_f3[5 * i:5 * i + 5]) / 5)
                #     stft_meanl.append(np.sum(stft_mean[5 * i:5 * i + 5]) / 5)
                #     stft_medianl.append(np.sum(stft_median[5 * i:5 * i + 5]) / 5)
                #     stft_maxl.append(np.sum(stft_max[5 * i:5 * i + 5]) / 5)
                #
                # if nbeat % 5 != 0:
                #     qrs_il.append(np.sum(qrs_i[int(nbeat / 5) * 5 - len(qrs_i):]) /
                #                   (len(qrs_i) - int(nbeat / 5) * 5))
                #     qt_il.append(np.sum(qt_i[int(nbeat / 5) * 5 - len(qt_i):]) /
                #                  (len(qt_i) - int(nbeat / 5) * 5))
                #     pq_il.append(np.sum(pq_i[int(nbeat / 5) * 5 - len(pq_i):]) /
                #                  (len(pq_i) - int(nbeat / 5) * 5))
                #     jt_il.append(np.sum(jt_i[int(nbeat / 5) * 5 - len(jt_i):]) /
                #                  (len(jt_i) - int(nbeat / 5) * 5))
                #     qrs_vl.append(np.sum(qrs_v[int(nbeat / 5) * 5 - len(qrs_v):]) /
                #                   (len(qrs_v) - int(nbeat / 5) * 5))
                #     t_vl.append(np.sum(t_v[int(nbeat / 5) * 5 - len(t_v):]) /
                #                 (len(t_v) - int(nbeat / 5) * 5))
                #     st20_vl.append(np.sum(st20_v[int(nbeat / 5) * 5 - len(st20_v):]) /
                #                    (len(st20_v) - int(nbeat / 5) * 5))
                #     qrs_argmaxl.append(np.sum(qrs_argmax[int(nbeat / 5) * 5 - len(qrs_argmax):]) /
                #                        (len(qrs_argmax) - int(nbeat / 5) * 5))
                #     qrs_argminl.append(np.sum(qrs_argmin[int(nbeat / 5) * 5 - len(qrs_argmin):]) /
                #                        (len(qrs_argmin) - int(nbeat / 5) * 5))
                #     qrs_maxl.append(np.sum(qrs_max[int(nbeat / 5) * 5 - len(qrs_max):]) /
                #                     (len(qrs_max) - int(nbeat / 5) * 5))
                #     qrs_minl.append(np.sum(qrs_min[int(nbeat / 5) * 5 - len(qrs_min):]) /
                #                     (len(qrs_min) - int(nbeat / 5) * 5))
                #     qrs_max_minl.append(np.sum(qrs_max_min[int(nbeat / 5) * 5 - len(qrs_max_min):]) /
                #                         (len(qrs_max_min) - int(nbeat / 5) * 5))
                #     qrs_argmax_argminl.append(np.sum(qrs_argmax_argmin[int(nbeat / 5) * 5 - len(qrs_argmax_argmin):]) /
                #                               (len(qrs_argmax_argmin) - int(nbeat / 5) * 5))
                #     qrs_al.append(np.sum(qrs_a[int(nbeat / 5) * 5 - len(qrs_a):]) /
                #                   (len(qrs_a) - int(nbeat / 5) * 5))
                #     qrst_al.append(np.sum(qrst_a[int(nbeat / 5) * 5 - len(qrst_a):]) /
                #                    (len(qrst_a) - int(nbeat / 5) * 5))
                #     pqrst_al.append(np.sum(pqrst_a[int(nbeat / 5) * 5 - len(pqrst_a):]) /
                #                     (len(pqrst_a) - int(nbeat / 5) * 5))
                #     qrs_qrst_ratiol.append(np.sum(qrs_qrst_ratio[int(nbeat / 5) * 5 - len(qrs_qrst_ratio):]) /
                #                            (len(qrs_qrst_ratio) - int(nbeat / 5) * 5))
                #     qrs_pqrst_ratiol.append(np.sum(qrs_pqrst_ratio[int(nbeat / 5) * 5 - len(qrs_pqrst_ratio):]) /
                #                             (len(qrs_pqrst_ratio) - int(nbeat / 5) * 5))
                #     qrst_pqrst_ratiol.append(np.sum(qrst_pqrst_ratio[int(nbeat / 5) * 5 - len(qrst_pqrst_ratio):]) /
                #                              (len(qrst_pqrst_ratio) - int(nbeat / 5) * 5))
                #     fft_f1l.append(np.sum(fft_f1[int(nbeat / 5) * 5 - len(fft_f1):]) /
                #                    (len(fft_f1) - int(nbeat / 5) * 5))
                #     fft_f2l.append(np.sum(fft_f2[int(nbeat / 5) * 5 - len(fft_f2):]) /
                #                    (len(fft_f2) - int(nbeat / 5) * 5))
                #     fft_f3l.append(np.sum(fft_f3[int(nbeat / 5) * 5 - len(fft_f3):]) /
                #                    (len(fft_f3) - int(nbeat / 5) * 5))
                #     stft_meanl.append(np.sum(stft_mean[int(nbeat / 5) * 5 - len(stft_mean):]) /
                #                       (len(stft_mean) - int(nbeat / 5) * 5))
                #     stft_medianl.append(np.sum(stft_median[int(nbeat / 5) * 5 - len(stft_median):]) /
                #                         (len(stft_median) - int(nbeat / 5) * 5))
                #     stft_maxl.append(np.sum(stft_max[int(nbeat / 5) * 5 - len(stft_max):]) /
                #                     (len(stft_max) - int(nbeat / 5) * 5))
                #
                # qrs_il = np.asarray(qrs_il).reshape([len(qrs_il), 1])
                # qt_il = np.asarray(qt_il).reshape([len(qt_il), 1])
                # pq_il = np.asarray(pq_il).reshape([len(pq_il), 1])
                # jt_il = np.asarray(jt_il).reshape([len(jt_il), 1])
                # qrs_vl = np.asarray(qrs_vl).reshape([len(qrs_vl), 1])
                # t_vl = np.asarray(t_vl).reshape([len(t_vl), 1])
                # st20_vl = np.asarray(st20_vl).reshape([len(st20_vl), 1])
                # qrs_argmaxl = np.asarray(qrs_argmaxl).reshape([len(qrs_argmaxl), 1])
                # qrs_argminl = np.asarray(qrs_argminl).reshape([len(qrs_argminl), 1])
                # qrs_maxl = np.asarray(qrs_maxl).reshape([len(qrs_maxl), 1])
                # qrs_minl = np.asarray(qrs_minl).reshape([len(qrs_minl), 1])
                # qrs_max_minl = np.asarray(qrs_max_minl).reshape([len(qrs_max_minl), 1])
                # qrs_argmax_argminl = np.asarray(qrs_argmax_argminl).reshape([len(qrs_argmax_argminl), 1])
                # qrs_al = np.asarray(qrs_al).reshape([len(qrs_al), 1])
                # qrst_al = np.asarray(qrst_al).reshape([len(qrst_al), 1])
                # pqrst_al = np.asarray(pqrst_al).reshape([len(pqrst_al), 1])
                # qrs_qrst_ratiol = np.asarray(qrs_qrst_ratiol).reshape([len(qrs_qrst_ratiol), 1])
                # qrs_pqrst_ratiol = np.asarray(qrs_pqrst_ratiol).reshape([len(qrs_pqrst_ratiol), 1])
                # qrst_pqrst_ratiol = np.asarray(qrst_pqrst_ratiol).reshape([len(qrst_pqrst_ratiol), 1])
                # fft_f1l = np.asarray(fft_f1l).reshape([len(fft_f1l), 1])
                # fft_f2l = np.asarray(fft_f2l).reshape([len(fft_f2l), 1])
                # fft_f3l = np.asarray(fft_f3l).reshape([len(fft_f3l), 1])
                # stft_meanl = np.asarray(stft_meanl).reshape([len(stft_meanl), 1])
                # stft_medianl = np.asarray(stft_medianl).reshape([len(stft_medianl), 1])
                # stft_maxl = np.asarray(stft_maxl).reshape([len(stft_maxl), 1])
                #
                # labell = np.ones([x_1.shape[0], 1]) if key in ['ST0-', 'ST1-',
                #                                                'ST0+', 'ST1+'] else np.zeros([x_1.shape[0], 1])
                # averaging by 5
                # a = np.concatenate((x_1, x_2, x_3, qrs_il, qt_il, pq_il, jt_il, qrs_vl, t_vl, st20_vl, qrs_argmaxl,
                #                     qrs_argminl, qrs_maxl, qrs_minl, qrs_max_minl, qrs_argmax_argminl, qrs_al,
                #                     qrst_al, pqrst_al, qrs_qrst_ratiol, qrs_pqrst_ratiol, qrst_pqrst_ratiol,
                #                     fft_f1l, fft_f2l, fft_f3l, stft_meanl, stft_medianl, stft_maxl, labell), axis=1)
                # no averaging

                """no averaging"""
                label = np.ones([pq_i.shape[0], 1]) if key in ['ST0-', 'ST1-',
                                                               'ST0+', 'ST1+'] else np.zeros([pq_i.shape[0], 1])
                qrs_i = qrs_i.reshape([qrs_i.shape[0], 1])
                qt_i = qt_i.reshape([qt_i.shape[0], 1])
                pq_i = pq_i.reshape([pq_i.shape[0], 1])
                jt_i = jt_i.reshape([jt_i.shape[0], 1])
                qrs_v = qrs_v.reshape([qrs_v.shape[0], 1])
                t_v = t_v.reshape([t_v.shape[0], 1])
                st20_v = st20_v.reshape([st20_v.shape[0], 1])
                qrs_max = qrs_max.reshape([qrs_max.shape[0], 1])
                qrs_min = qrs_min.reshape([qrs_min.shape[0], 1])
                qrs_max_min = qrs_max_min.reshape([qrs_max_min.shape[0], 1])
                qrs_argmax_argmin = qrs_argmax_argmin.reshape([qrs_argmax_argmin.shape[0], 1])
                qrs_qrst_ratio = qrs_qrst_ratio.reshape([qrs_qrst_ratio.shape[0], 1])
                qrs_pqrst_ratio = qrs_pqrst_ratio.reshape([qrs_pqrst_ratio.shape[0], 1])
                qrst_pqrst_ratio = qrst_pqrst_ratio.reshape([qrst_pqrst_ratio.shape[0], 1])

                # b = np.concatenate((x_1, x_2, x_3, qrs_i, qt_i, pq_i, jt_i, qrs_v, t_v, st20_v, qrs_argmax,
                #                     qrs_argmin, qrs_max, qrs_min, qrs_max_min, qrs_argmax_argmin, qrs_a,
                #                     qrst_a, pqrst_a, qrs_qrst_ratio, qrs_pqrst_ratio, qrst_pqrst_ratio,
                #                     fft_f1, fft_f2, fft_f3, stft_mean, stft_median, stft_max, label), axis=1)
                # selected features
                b = np.concatenate((x_2, x_3, pq_i, qrs_v, st20_v, qrs_argmin, qrs_min, qrs_max_min, qrst_a,
                                    fft_f1, fft_f2, fft_f3, stft_mean, stft_median, stft_max, label), axis=1)

                if key in ['ST0-', 'ST1-', 'ST0+', 'ST1+']:
                    c_st += b.shape[0]
                else:
                    c_n += b.shape[0]

                print("      Generated numpy array has shape:", b.shape)

                train_dict[atr][key] = b.tolist()

                # np.save('./trainingsets/{}_{}.npy'.format('e0606', key), b)

                # X = b[:, :15]
                # y = b[:, 15]
                # clf = joblib.load('./clf_model/rf_clf_2018_11_08.model')
                # mms = MinMaxScaler()
                # X = mms.fit_transform(X)
                # prediction = clf.predict(X)
                # score = clf.score(X, y)
                # for i in range(R_located.shape[0]):
                #     p = 'ST' if prediction[i] == 1 else 'N'
                #     if key in ['ST0-', 'ST1-', 'ST0+', 'ST1+']:
                #         plt.text(R_located[i], signal[R_located[i]], "{}_{}\nP_{}".format(i, 'ST', p))
                #     else:
                #         plt.text(R_located[i], signal[R_located[i]], "{}_{}\nP_{}".format(i, 'N', p))
                # plt.title('QRSPT Delineation with accuracy {}'.format(score))
                # plt.show()

            except:
                print("---Record {}, Type {} failed---".format(atr, key))

print("All finished! Total ST samples {} and normal samples {}".format(c_st, c_n))

with open('./trainingsets/edb_train_sets.json', 'w') as outfile:
    json.dump(train_dict, outfile)

# delete the redundancy points and compensate the missing points for Mj3
# num2 = 1
# while num2 != 0:
#     num2 = 0
#     R = np.nonzero(countR)[0]
#     R_R = R[1:] - R[:-1]
#     R_R_mean = np.mean(R_R)
#     for i in range(1, R.shape[0]):
#         if R[i] - R[i - 1] <= 0.4 * R_R_mean:
#             num2 += 1
#             if signal[R[i]] > signal[R[i - 1]]:
#                 countR[R[i - 1]] = 0
#             else:
#                 countR[R[i]] = 0
#
# num1 = 2
# while num1 > 0:
#     k = 0
#     num1 -= 1
#     R = np.nonzero(countR)[0]
#     R_R = R[1:] - R[:-1]
#     R_R_mean = np.mean(R_R)
#     for i in range(1, R.shape[0]):
#         if R[i] - R[i - 1] > 1.6 * R_R_mean:
#             Mj_adjust = w_peak[3, R[i - 1] + 80:R[i] - 80]
#             points2 = (R[i] - 80) - (R[i - 1] + 80) + 1
#
#             adjust_pos = Mj_adjust * (Mj_adjust > 0)
#             adjust_pos = (adjust_pos > thpos_i / 4)
#             adjust_neg = Mj_adjust * (Mj_adjust < 0)
#             adjust_neg = -1 * (adjust_neg < thneg_i / 5)
#             interval4 = adjust_pos + adjust_neg
#             loca3 = np.nonzero(interval4)[0]
#             diff2 = interval4[loca3[:-1]] - interval4[loca3[1:]]
#
#             loca4 = np.where(diff2 == -2)[0]
#             interval3 = np.zeros(points2)
#             for j in range(loca4.shape[0]):
#                 interval3[loca3[loca4[j]]] = interval4[loca3[loca4[j]]]
#                 interval3[loca3[loca4[j] + 1]] = interval4[loca3[loca4[j] + 1]]
#             mark4 = 0
#             mark5 = 0
#             mark6 = 0
#
#             while k < points2 - 1:
#                 if interval3[k] == -1:
#                     mark4 = k
#                     k += 1
#                     while k < points2 - 1 and interval3[k] == 0:
#                         k += 1
#                     mark5 = k
#                     mark6 = int(round((abs(Mj_adjust[mark5]) * (mark4 + 1) + (mark5 + 1) * abs(Mj_adjust[mark4])) /
#                                       (abs(Mj_adjust[mark5]) + abs(Mj_adjust[mark4]))))
#                     countR[R[i - 1] + 80 + mark6 - 10] = 1
#                     k += 60
#                 k += 1
# """find P and T peaks, P and T waves are more salient in Mj4"""
# Mj4_pos = Mj4 * (Mj4 > 0)
# Mj4_thpos = (max(Mj4_pos[0:round(points / 4)]) +
#              max(Mj4_pos[round(points / 4):2 * round(points / 4)]) +
#              max(Mj4_pos[2 * round(points / 4):3 * round(points / 4)]) +
#              max(Mj4_pos[3 * round(points / 4):4 * round(points / 4)])) / 4
# Mj4_pos = 1 * (Mj4_pos > Mj4_thpos / 3)
# Mj4_neg = Mj4 * (Mj4 < 0)
# Mj4_thneg = (min(Mj4_neg[0:round(points / 4)]) +
#              min(Mj4_neg[round(points / 4):2 * round(points / 4)]) +
#              min(Mj4_neg[2 * round(points / 4):3 * round(points / 4)]) +
#              min(Mj4_neg[3 * round(points / 4):4 * round(points / 4)])) / 4
# Mj4_neg = -1 * (Mj4_neg < Mj4_thneg / 4)
# Mj4_interval = Mj4_pos + Mj4_neg
# Mj4_loca = np.nonzero(Mj4_interval)[0]
# Mj4_interval2 = np.zeros(points)
# Mj4_diff = []
# for i in range(0, Mj4_loca.shape[0] - 1):
#     if abs(Mj4_loca[i] - Mj4_loca[i + 1]) < 80:
#         Mj4_diff.append(Mj4_interval[Mj4_loca[i]] - Mj4_interval[Mj4_loca[i + 1]])
#     else:
#         Mj4_diff.append(0)
# Mj4_diff = np.array(Mj4_diff)
# Mj4_loca2 = np.where(Mj4_diff == -2)[0]
# Mj4_interval2[Mj4_loca[Mj4_loca2]] = Mj4_interval[Mj4_loca[Mj4_loca2]]
# Mj4_interval2[Mj4_loca[Mj4_loca2 + 1]] = Mj4_interval[Mj4_loca[Mj4_loca2 + 1]]
#
# mark7 = 0
# mark8 = 0
# mark9 = 0
# Mj4_countR = np.zeros(1)
# Mj4_countQ = np.zeros(1)
# Mj4_countS = np.zeros(1)
# l = 0
# flag = 0
# while l < points - 1:
#     if Mj4_interval2[l] == -1:
#         mark7 = l
#         l += 1
#         while l < points - 1 and Mj4_interval2[l] == 0:
#             l += 1
#         mark8 = l
#         mark9 = int(round((abs(Mj4[mark8]) * (mark7 + 1) + (mark8 + 1) * abs(Mj4[mark7])) /
#                           (abs(Mj4[mark8]) + abs(Mj4[mark7]))))
#         Mj4_countR = np.concatenate((Mj4_countR, np.zeros(mark9 - 13 - Mj4_countR.shape[0]), np.ones(1)))
#         # Mj4_countQ = np.concatenate((Mj4_countQ, np.zeros(mark7 - 12 - Mj4_countQ.shape[0]), -1 * np.ones(1)))
#         # Mj4_countS = np.concatenate((Mj4_countS, np.zeros(mark8 - 12 - Mj4_countS.shape[0]), -1 * np.ones(1)))
#         flag = 1
#
#         kqs = mark9 - 13
#         mark_q = 0
#         while kqs > 0 and mark_q < 1:
#             if Mj4[kqs] != 0:
#                 mark_q += 1
#             kqs -= 1
#         Mj4_countQ = np.concatenate((Mj4_countQ, np.zeros(kqs - Mj4_countQ.shape[0]), -1 * np.ones(1)))
#
#         kqs = mark9 - 13
#         mark_s = 0
#         while kqs < points - 1 and mark_s < 1:
#             if Mj4[kqs] != 0:
#                 mark_s += 1
#             kqs += 1
#         Mj4_countS = np.concatenate((Mj4_countS, np.zeros(kqs - Mj4_countS.shape[0]), -1 * np.ones(1)))
#
#     if flag == 1:
#         l += 100
#         flag = 0
#     else:
#         l += 1
#
# # plt.figure(4)
# # plt.plot(Mj4_interval2)
# # plt.plot(Mj4_countR, 'r')
# # plt.plot(Mj4_countQ, 'k')
# # plt.plot(Mj4_countS, 'k')
# # plt.show()
#
# """delete the redundancy points and compensate the missing points for Mj4"""
# num4 = 1
# while num4 != 0:
#     num4 = 0
#     R = np.nonzero(Mj4_countR)[0]
#     R_R = R[1:] - R[:-1]
#     R_R_mean = np.mean(R_R)
#     for i in range(1, R.shape[0]):
#         if R[i] - R[i - 1] <= 0.4 * R_R_mean:
#             num4 += 1
#             if signal[R[i]] > signal[R[i - 1]]:
#                 Mj4_countR[R[i - 1]] = 0
#             else:
#                 Mj4_countR[R[i]] = 0
#
# num3 = 2
# while num3 > 0:
#     k = 0
#     num3 -= 1
#     R = np.nonzero(Mj4_countR)[0]
#     R_R = R[1:] - R[:-1]
#     R_R_mean = np.mean(R_R)
#     for i in range(1, R.shape[0]):
#         if R[i] - R[i - 1] > 1.6 * R_R_mean:
#             Mj4_adjust = w_peak[4, R[i - 1] + 80:R[i] - 80]
#             points2 = (R[i] - 80) - (R[i - 1] + 80) + 1
#
#             adjust_pos = Mj4_adjust * (Mj4_adjust > 0)
#             adjust_pos = (adjust_pos > Mj4_thpos / 4)
#             adjust_neg = Mj4_adjust * (Mj4_adjust < 0)
#             adjust_neg = -1 * (adjust_neg < Mj4_thneg / 5)
#             Mj4_interval4 = adjust_pos + adjust_neg
#             Mj4_loca3 = np.nonzero(Mj4_interval4)[0]
#             Mj4_diff2 = Mj4_interval4[Mj4_loca3[:-1]] - Mj4_interval4[Mj4_loca3[1:]]
#
#             Mj4_loca4 = np.where(Mj4_diff2 == -2)[0]
#             Mj4_interval3 = np.zeros(points2)
#             for j in range(Mj4_loca4.shape[0]):
#                 Mj4_interval3[Mj4_loca3[Mj4_loca4[j]]] = Mj4_interval4[Mj4_loca3[Mj4_loca4[j]]]
#                 Mj4_interval3[Mj4_loca3[Mj4_loca4[j] + 1]] = Mj4_interval4[Mj4_loca3[Mj4_loca4[j] + 1]]
#             mark4 = 0
#             mark5 = 0
#             mark6 = 0
#
#             while k < points2 - 1:
#                 if Mj4_interval3[k] == -1:
#                     mark4 = k
#                     k += 1
#                     while k < points2 - 1 and Mj4_interval3[k] == 0:
#                         k += 1
#                     mark5 = k
#                     mark6 = int(round((abs(Mj4_adjust[mark5]) * (mark4 + 1) + (mark5 + 1) * abs(Mj4_adjust[mark4])) /
#                                       (abs(Mj4_adjust[mark5]) + abs(Mj4_adjust[mark4]))))
#                     Mj4_countR[R[i - 1] + 80 + mark6 - 10] = 1
#                     k += 60
#                 k += 1
#
# # plt.figure(5)
# # plt.plot(signal)
# # plt.plot(Mj4_countR, 'r')
# # plt.plot(Mj4_countQ, 'k')
# # plt.plot(Mj4_countS, 'k')
# # plt.show()
#
# R_located = np.nonzero(Mj4_countR)[0]
# Q_located = np.nonzero(Mj4_countQ)[0]
# S_located = np.nonzero(Mj4_countS)[0]
# Mj4_countP = np.zeros(1)
# Mj4_countT = np.zeros(1)
# countP = np.zeros(1)
# countP_begin = np.zeros(1)
# countP_end = np.zeros(1)
# countT = np.zeros(1)
# countT_begin = np.zeros(1)
# countT_end = np.zeros(1)
#
# """detect P wave"""
# window_size = 100
# mark10 = 0
# for i in range(1, R_located.shape[0]):
#     flag = 0
#     mark10 = 0
#     R_R_interval = R_located[i] - R_located[i - 1]
#     for j in range(0, int(R_R_interval * 2 / 3), 5):
#         window_end = Q_located[i] - j
#         window_begin = window_end - window_size
#         if window_begin < R_located[i - 1] + R_R_interval / 3:
#             break
#         window_max = np.max(Mj4[window_begin:window_end])
#         window_max_idx = np.argmax(Mj4[window_begin:window_end])
#         window_min = np.min(Mj4[window_begin:window_end])
#         window_min_idx = np.argmin(Mj4[window_begin:window_end])
#         if window_min_idx < window_max_idx and (window_max_idx - window_min_idx) < window_size * 2 / 3 \
#            and window_max > 0.01 and window_min < -0.1:
#             flag = 1
#             mark10 = int(round((window_max_idx + window_min_idx) / 2 + window_begin))
#             Mj4_countP = np.concatenate((Mj4_countP, np.zeros(mark10 - Mj4_countP.shape[0]), np.ones(1)))
#             countP = np.concatenate((countP, np.zeros(mark10 - countP.shape[0]), -1 * np.ones(1)))
#             countP_begin = np.concatenate((countP_begin, np.zeros(window_begin + window_min_idx -
#                                                                   countP_begin.shape[0]), -1 * np.ones(1)))
#             countP_end = np.concatenate((countP_end, np.zeros(window_begin + window_max_idx - countP_end.shape[0]),
#                                          -1 * np.ones(1)))
#         if flag == 1:
#             break
#     if mark10 == 0 and flag == 0:
#         mark10 = int(round(R_located[i] - R_R_interval / 3))
#         countP = np.concatenate((countP, np.zeros(mark10 - countP.shape[0]), -1 * np.ones(1)))
#
# """detect T wave"""
# window_size_Q = 90
# mark11 = 0
# for i in range(0, R_located.shape[0] - 1):
#     flag = 0
#     mark11 = 0
#     R_R_interval = R_located[i + 1] - R_located[i]
#     for j in range(0, int(R_R_interval * 2 / 3), 5):
#         window_begin = S_located[i] + j
#         window_end = window_begin + window_size_Q
#         if window_end > R_located[i + 1] - R_R_interval / 4:
#             break
#         window_max = np.max(Mj4[window_begin:window_end])
#         window_max_idx = np.argmax(Mj4[window_begin:window_end])
#         window_min = np.min(Mj4[window_begin:window_end])
#         window_min_idx = np.argmin(Mj4[window_begin:window_end])
#         if window_min_idx < window_max_idx and (window_max_idx - window_min_idx) < window_size_Q \
#            and window_max > 0.1 and window_min < -0.1:
#             flag = 1
#             mark11 = int(round((window_max_idx + window_min_idx) / 2 + window_begin))
#             Mj4_countT = np.concatenate((Mj4_countT, np.zeros(mark11 - Mj4_countT.shape[0]), np.ones(1)))
#             countT = np.concatenate((countT, np.zeros(mark11 - countT.shape[0]), -1 * np.ones(1)))
#             countT_begin = np.concatenate((countT_begin, np.zeros(window_begin + window_min_idx -
#                                                                   countT_begin.shape[0]), -1 * np.ones(1)))
#             countT_end = np.concatenate((countT_end, np.zeros(window_begin + window_max_idx - countT_end.shape[0]),
#                                          -1 * np.ones(1)))
#         if flag == 1:
#             break
#     if mark11 == 0 and flag == 0:
#         mark11 = int(round(R_located[i] - R_R_interval / 3))
#         countT = np.concatenate((countT, np.zeros(mark11 - countT.shape[0]), -2 * np.ones(1)))
#
# # plt.figure(3)
# # plt.plot(signal)
# # plt.plot(countR, 'r')
# # plt.plot(countQ, 'k')
# # plt.plot(countS, 'g')
# # # plt.plot(countP, 'r')
# # plt.plot(countT, 'r')
# # for i in range(Rnum):
# #     if R_results[i] == 0:
# #         break
# #     plt.plot(R_results[i], signal[R_results[i]], marker='o', color='g')
# # plt.show()
