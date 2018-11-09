from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pywt
import wfdb
from wfdb import processing
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

sampling_rate = 250  # sampling rate is 250Hz
k = int(np.ceil(np.log2(sampling_rate)))  # decomposition level
ROOT_PATH = '/Users/sumuzhao/ETHZ/Semester_project_2/'


def generate_atr_list(atr_path, is_edb=True):
    """generate the effective annotation file list"""
    atr_list = []
    for dirpath, _, filenames in os.walk(atr_path):
        for filename in filenames:
            basename = os.path.splitext(filename)[0]
            if is_edb:
                if filename.endswith('.atr') and basename not in ['e0133', 'e0155', 'e0509', 'e0611', 'e0163',
                                                                  'e0405', 'e0409', 'e0704']:
                    atr_list.append(basename)
            else:
                if filename.endswith('.atr'):
                    atr_list.append(basename)
    atr_list.sort(key=lambda x: int(os.path.splitext(x)[0][1:]))
    return atr_list


def pick_s_symbols(atr_name, is_edb=True, same_st_episodes='stb'):
    """pick up all the effective 's' symbols"""

    if is_edb:
        ann = wfdb.rdann('./edb_records/{}'.format(atr_name), 'atr')
        symbol_list = ann.symbol
        sample_list = ann.sample
        aux_list = ann.aux_note
        sym_sam_list = list(zip(sample_list, aux_list))
        st_idx = []
        for j in range(len(symbol_list)):
            if symbol_list[j] == 's':
                st_idx.append(sym_sam_list[j])

    else:

        ann = wfdb.rdann('./ltst_records/{}'.format(atr_name), same_st_episodes)
        symbol_list = ann.symbol
        sample_list = ann.sample
        aux_list = ann.aux_note
        sym_sam_list = list(zip(sample_list, aux_list))
        st_idx = []
        for j in range(len(symbol_list)):
            if symbol_list[j] == 's':
                st_idx.append(sym_sam_list[j])

    return st_idx


def pick_st_pairs(st_idx, is_edb=True):
    """pick up all the 'ST' symbol pairs for each episodes"""

    if is_edb:

        dict = {'ST0+': [], 'ST1+': [], 'ST0-': [], 'ST1-': []}

        for i in range(len(st_idx) - 1):
            for j in range(i + 1, len(st_idx)):
                if st_idx[i][1] == '(ST0+\x00' and st_idx[j][1] == 'ST0+)\x00':
                    dict['ST0+'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break
                elif st_idx[i][1] == '(ST1+\x00' and st_idx[j][1] == 'ST1+)\x00':
                    dict['ST1+'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break
                elif st_idx[i][1] == '(ST0-\x00' and st_idx[j][1] == 'ST0-)\x00':
                    dict['ST0-'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break
                elif st_idx[i][1] == '(ST1-\x00' and st_idx[j][1] == 'ST1-)\x00':
                    dict['ST1-'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break

    else:

        dict = {'ST0': [], 'ST1': [], 'ST2': []}

        for i in range(len(st_idx) - 1):
            for j in range(i + 1, len(st_idx)):
                if (st_idx[i][1][:4] == '(st0' and st_idx[j][1][:3] == 'st0') or \
                   (st_idx[i][1][:6] == '(rtst0' and st_idx[j][1][:5] == 'rtst0'):
                    dict['ST0'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break
                elif (st_idx[i][1][:4] == '(st1' and st_idx[j][1][:3] == 'st1') or \
                     (st_idx[i][1][:6] == '(rtst1' and st_idx[j][1][:5] == 'rtst1'):
                    dict['ST1'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break
                elif (st_idx[i][1][:4] == '(st2' and st_idx[j][1][:3] == 'st2') or \
                     (st_idx[i][1][:6] == '(rtst2' and st_idx[j][1][:5] == 'rtst2'):
                    dict['ST2'].append((int(st_idx[i][0]), int(st_idx[j][0])))
                    break

    for key in list(dict.keys()):
        if not dict[key]:
            dict.pop(key)

    return dict


def create_st_dictionary(record_path='./edb_records', is_edb=True, same_st_episodes='stb'):
    """
    create a dictionary for all the records
    format: is_edb=True,
            {'e0103':
                {'ST0+': [(start, end), ...],
                 'ST1+': [(start, end), ...],
                 'ST0-': [(start, end), ...],
                 'ST1-': [(start, end), ...]},
             'e0104':
                {},
              ...
            }

            is_edb=False
            {'s20011':
                {'ST0': [(start, end), ...],
                 'ST1': [(start, end), ...],
                 'ST2': [(start, end), ...]},
             's20011':
                {},
              ...
            }

    """
    dict = {}
    atr_list = generate_atr_list(record_path, is_edb)
    for i in range(len(atr_list)):
        st_idx = pick_s_symbols(atr_list[i], is_edb, same_st_episodes)
        st_dict= pick_st_pairs(st_idx, is_edb)
        print(i, st_dict)
        if st_dict:
            dict[atr_list[i]] = st_dict
        else:
            print("Drop empty record {}".format(atr_list[i]))

    return atr_list, dict


def st_histogram(atr_list, st_dict):
    st0_p = 0
    st0_n = 0
    st1_p = 0
    st1_n = 0

    for atr in atr_list:
        for key in list(st_dict[atr].keys()):
            if key == 'ST0+':
                st0_p += 1
            elif key == 'ST0-':
                st0_n += 1
            elif key == 'ST1+':
                st1_p += 1
            elif key == 'ST1-':
                st1_n += 1
    plt.bar(range(4), [st0_p, st0_n, st1_p, st1_n], color='lightsteelblue')
    plt.xlabel("ST episode categories")
    plt.ylabel("Number of ST episodes")
    for x, y in zip(range(4), [st0_p, st0_n, st1_p, st1_n]):
        plt.text(x, y, y, ha='center', va='bottom')
    plt.xticks(range(4), ['ST0+', 'ST0-', 'ST1+', 'ST1-'])
    plt.show()


def smooth(y, box_pts):
    """smooth function"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_normal_part(atr_name, dict, data_path='./edb_records/'):
    """
    get the normal parts for each record (eliminate the ST episodes)
    return normal_dict: dict{}
                        format: {'N0': [np.array()],
                                 'N1': [np.array()],
                                 'N2': [np.array()]}
    """
    normal_signal_dict = {'N0': [], 'N1': [], 'N2': []}
    parse_list = parse_st_sub_dict(atr_name, dict)
    signal, _ = wfdb.rdsamp(data_path + atr_name)

    st0_length = 0
    st1_length = 0
    st2_length = 0
    for pl in parse_list:
        part_list = dict[atr_name][pl[0]]

        for part in part_list:
            if pl[1] == 0:
                st0_length += part[1] - part[0]
            elif pl[1] == 1:
                st1_length += part[1] - part[0]
            else:
                st2_length += part[1] - part[0]

            signal[part[0] - 1000:part[1] + 1000, pl[1]] = 0

    signal0 = signal[:, 0]
    signal1 = signal[:, 1]
    signal0 = signal0[signal0 != 0]
    signal1 = signal1[signal1 != 0]

    if st0_length != 0:
        normal_signal_dict['N0'].append(signal0[0:st0_length])
    if st1_length != 0:
        normal_signal_dict['N1'].append(signal1[0:st1_length])

    if atr_name[1] == '3':
        signal2 = signal[:, 2]
        signal2 = signal2[signal2 != 0]
        if st2_length != 0:
            normal_signal_dict['N2'].append(signal2[0:st2_length])

    return normal_signal_dict


def load_st_signal(atr_name, dict, data_path='./edb_records/', is_edb=True):
    """
    load st episodes signals
    return st_signal_dict: dict{}
                        format: {'ST0-': [np.array()],
                                 'ST1-': [np.array()],
                                 'ST0+': [np.array()],
                                 'ST1+': [np.array()],
                                 'ST0': [np.array()],
                                 'ST1': [np.array()]}
    """

    if is_edb:
        st_signal_dict = {'ST0-': [], 'ST1-': [], 'ST0+': [], 'ST1+': []}
    else:
        st_signal_dict = {'ST0': [], 'ST1': [], 'ST2': []}

    parse_list = parse_st_sub_dict(atr_name, dict)

    for pl in parse_list:
        if dict[atr_name][pl[0]]:
            for st_pair in dict[atr_name][pl[0]]:
                st_signal, _ = wfdb.rdsamp(data_path + atr_name, sampfrom=st_pair[0], sampto=st_pair[1])
                st_signal_dict[pl[0]].append(st_signal[:, pl[1]])
            signal_concat = st_signal_dict[pl[0]][0]
            for i in st_signal_dict[pl[0]][1:]:
                signal_concat = np.concatenate((signal_concat, i))
            if signal_concat.shape[0] % 2 == 1:
                signal_concat = signal_concat[0:signal_concat.shape[0] - 1]
            st_signal_dict[pl[0]] = [signal_concat]

    return st_signal_dict


def parse_st_sub_dict(atr_name, dict):
    """
    parse each record, return signal channel and whether it is elevation or depression
    return parse_list: [(st episode category, signal channel, elevation or depression)]
    """
    parse_list = []
    sub_dict = dict[atr_name]
    key = list(sub_dict.keys())
    for k in key:

        if k in ['ST0+', 'ST0-', 'ST0']:
            channel = 0
        elif k in ['ST1+', 'ST1-', 'ST1']:
            channel = 1
        else:
            channel = 2

        if k in ['ST0+', 'ST1+']:
            is_elevation = True
        else:
            is_elevation = False
        parse_list.append((k, channel, is_elevation))

    return parse_list


def remove_baseline(signal, wavelet='bior2.6', is_plot=False):
    """
    Removal of baseline wandering using wavelet
    wavelet: bior2.6
    level: 8
    """

    A8, D8, D7, D6, D5, D4, D3, D2, D1 = pywt.wavedec(signal, wavelet=pywt.Wavelet(wavelet), level=8)
    A8 = np.zeros_like(A8[0])  # low frequency info
    RA7 = pywt.idwt(A8, D8[0], wavelet)
    RA6 = pywt.idwt(RA7[0:len(D7[0])], D7[0], wavelet)
    RA5 = pywt.idwt(RA6[0:len(D6[0])], D6[0], wavelet)
    RA4 = pywt.idwt(RA5[0:len(D5[0])], D5[0], wavelet)
    RA3 = pywt.idwt(RA4[0:len(D4[0])], D4[0], wavelet)
    RA2 = pywt.idwt(RA3[0:len(D3[0])], D3[0], wavelet)
    D2 = np.zeros_like(D2[0])  # high frequency noise
    RA1 = pywt.idwt(RA2[0:len(D2)], D2, wavelet)
    D1 = np.zeros_like(D1[0])  # high frequency noise
    DenoisingSignal = pywt.idwt(RA1[0:len(D1)], D1, wavelet)

    if is_plot:
        plt.plot(signal[0], 'b')
        plt.plot(DenoisingSignal, 'g')
        plt.show()

    return DenoisingSignal


def binary_spline_wavelet_filter(swa, swd, signal, points, level=4, is_plot=False):
    """
    Binary spline wavelet filtering, level: 4,
    low-pass filter：[1/4 3/4 3/4 1/4], high-pass filter：[-1/4 -3/4 3/4 1/4]
    """

    for i in range(0, points - 3):
        swa[0, i + 3] = 1 / 4 * signal[i + 3 - 2 ** 0 * 0] + \
                        3 / 4 * signal[i + 3 - 2 ** 0 * 1] + \
                        3 / 4 * signal[i + 3 - 2 ** 0 * 2] + \
                        1 / 4 * signal[i + 3 - 2 ** 0 * 3]
        swd[0, i + 3] = - 1 / 4 * signal[i + 3 - 2 ** 0 * 0] - \
                        3 / 4 * signal[i + 3 - 2 ** 0 * 1] + \
                        3 / 4 * signal[i + 3 - 2 ** 0 * 2] + \
                        1 / 4 * signal[i + 3 - 2 ** 0 * 3]
    for j in range(1, level):
        for i in range(0, points - 24):
            swa[j, i + 24] = 1 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 0] \
                             + 3 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 1] \
                             + 3 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 2] \
                             + 1 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 3]
            swd[j, i + 24] = - 1 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 0] \
                             - 3 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 1] \
                             + 3 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 2] \
                             + 1 / 4 * swa[j - 1, i + 24 - 2 ** (j - 1) * 3]

    if is_plot:
        # draw the original signal and all the approximation and details coefficients.
        plt.figure(1)
        ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2)
        ax1.plot(signal)
        ax1.set_title("Original signal")
        for i in range(level):
            ax_a = plt.subplot2grid((5, 2), (i + 1, 0))
            ax_a.plot(swa[i, :])
            ax_d = plt.subplot2grid((5, 2), (i + 1, 1))
            ax_d.plot(swd[i, :])
        plt.show()

    return swa, swd


def get_maximal_value_pairs(swd, points, level, is_plot=False):
    """calculate the positive and negative maximal value pairs"""

    pos_idx = np.zeros_like(swd, dtype=np.int32)
    neg_idx = np.zeros_like(swd, dtype=np.int32)
    pos_w = swd * (swd > 0)
    pos_dw = ((pos_w[:, 0:points - 1] - pos_w[:, 1:points]) < 0)
    pos_idx[:, 1:points - 1] = ((pos_dw[:, 0:points - 2] - pos_dw[:, 1:points - 1]) > 0)
    neg_w = swd * (swd < 0)
    neg_dw = ((neg_w[:, 0:points - 1] - neg_w[:, 1:points]) > 0)
    neg_idx[:, 1:points - 1] = ((neg_dw[:, 0:points - 2] - neg_dw[:, 1:points - 1]) > 0)
    pos_neg_idx = np.bitwise_or(pos_idx, neg_idx)
    pos_neg_idx[:, 0] = 1
    pos_neg_idx[:, points - 1] = 1
    w_peak = pos_neg_idx * swd
    w_peak[:, 0] += 1e-10
    w_peak[:, points - 1] += 1e-10

    if is_plot:
        # ECG信号在j=1,2,3,4尺度下的小波系数的模极大值点
        plt.figure(2)
        for i in range(level):
            ax = plt.subplot2grid((4, 1), (i, 0))
            ax.plot(w_peak[i, :])
            ax.set_title('level_{}'.format(i))
        plt.show()

    return w_peak


def get_difference(w_peak, points, level):
    """get difference to determine the locations of maximal values"""

    sig = w_peak[level - 1, :]
    pos_i = sig * (sig > 0)
    thpos_i = (max(pos_i[0:round(points / 4)]) +
                 max(pos_i[round(points / 4):2 * round(points / 4)]) +
                 max(pos_i[2 * round(points / 4):3 * round(points / 4)]) +
                 max(pos_i[3 * round(points / 4):4 * round(points / 4)])) / 4
    pos_i = 1 * (pos_i > thpos_i / 3)
    neg_i = sig * (sig < 0)
    thneg_i = (min(neg_i[0:round(points / 4)]) +
                 min(neg_i[round(points / 4):2 * round(points / 4)]) +
                 min(neg_i[2 * round(points / 4):3 * round(points / 4)]) +
                 min(neg_i[3 * round(points / 4):4 * round(points / 4)])) / 4
    neg_i = - 1 * (neg_i < thneg_i / 4)

    interval = pos_i + neg_i
    loca = np.where(interval != 0)[0]

    diff = []
    for i in range(0, loca.shape[0] - 1):
        if abs(loca[i] - loca[i + 1]) < 80:
            diff.append(interval[loca[i]] - interval[loca[i + 1]])
        else:
            diff.append(0)

    return interval, loca, diff


def qrs_delineation(w_peak, points, interval2):
    """detect Q, R, S, onset, offset for each QRS complex"""

    countR = np.zeros(1, dtype=np.int32)
    countQ = np.zeros(1, dtype=np.int32)
    countS = np.zeros(1, dtype=np.int32)
    count_onset = np.zeros(1, dtype=np.int32)
    count_offset = np.zeros(1, dtype=np.int32)

    Mj1 = w_peak[0, :]
    Mj3 = w_peak[2, :]

    i = 0
    Rnum = 0
    R_results = []
    while i < points - 1:
        if interval2[i] == -1:
            mark1 = i
            i += 1
            while i < points - 1 and interval2[i] == 0:
                i += 1
            mark2 = i
            mark3 = int(round((abs(Mj3[mark2]) * (mark1 + 1) + (mark2 + 1) * abs(Mj3[mark1])) /
                              (abs(Mj3[mark2]) + abs(Mj3[mark1]))))
            R_results.append(mark3 - 5)
            countR = np.concatenate((countR, np.zeros(mark3 - 5 - countR.shape[0]), np.ones(1)))

            kqs = mark3 - 5
            mark_q = 0
            while kqs > 0 and mark_q < 3:
                if Mj1[kqs] != 0:
                    mark_q += 1
                kqs -= 1
            countQ = np.concatenate((countQ, np.zeros(kqs - countQ.shape[0]), -1 * np.ones(1)))

            kqs = mark3 - 5
            mark_s = 0
            while kqs < points - 1 and mark_s < 3:
                if Mj1[kqs] != 0:
                    mark_s += 1
                kqs += 1
            countS = np.concatenate((countS, np.zeros(kqs - countS.shape[0]), -1 * np.ones(1)))

            kqs = mark3 - 5
            mark_q = 0
            while kqs > 0 and mark_q < 5:
                if Mj1[kqs] != 0:
                    mark_q += 1
                kqs -= 1
            count_onset = np.concatenate((count_onset, np.zeros(kqs - count_onset.shape[0]), -1 * np.ones(1)))

            kqs = mark3 - 5
            mark_s = 0
            while kqs < points - 1 and mark_s < 5:
                if Mj1[kqs] != 0:
                    mark_s += 1
                kqs += 1
            count_offset = np.concatenate((count_offset, np.zeros(kqs - count_offset.shape[0]), -1 * np.ones(1)))

            i += 60
            Rnum += 1
        i += 1

    R_located = np.nonzero(countR)[0]
    S_located = np.nonzero(countS)[0]
    Q_located = np.nonzero(countQ)[0]
    onset_located = np.nonzero(count_onset)[0]
    offset_located = np.nonzero(count_offset)[0]

    return Q_located, R_located, S_located, onset_located, offset_located


def get_fecg(signal, wavelet=pywt.Wavelet('db8')):
    """calculate the flatted ecg data"""
    # get the baseline and fecg
    coeffs_1 = pywt.wavedec(signal, wavelet, 'periodization', k)
    new_coeffs_1 = []
    new_coeffs_1.append(coeffs_1[0])
    for coeff in coeffs_1[1:]:
        new_coeffs_1.append(np.zeros(coeff.shape))
    baseline = pywt.waverec(new_coeffs_1, wavelet, 'periodization')

    if len(signal) > len(baseline):
        signal = signal[0:len(baseline)]
    elif len(signal) < len(baseline):
        baseline = baseline[0:len(signal)]

    fecg = signal - baseline
    plt.plot(fecg, 'g-')
    return fecg


def find_r_peak(fecg, wavelet=pywt.Wavelet('db8')):
    coeffs_2 = pywt.wavedec(fecg, wavelet, 'periodization', k)
    new_coeffs_2 = []
    for i in range(k + 1):
        new_coeffs_2.append(np.zeros(coeffs_2[i].shape))

    score = []
    for i in range(k, 1, -1):
        new_coeffs_2[i] = coeffs_2[i]
        pulse = pywt.waverec(new_coeffs_2, wavelet, 'periodization')
        new_coeffs_2[i] = np.zeros(coeffs_2[i].shape)
        sum_pulse = np.sum(np.abs(pulse))
        score.append(np.abs(np.sum(fecg * np.abs(pulse) / sum_pulse)))

    score_diff = [score[i] - score[i + 1] for i in range(1, len(score) - 1)]
    chosen_scale = int(np.argmax(score_diff) + 2)
    new_coeffs_2[-chosen_scale] = coeffs_2[-chosen_scale]
    pulse = pywt.waverec(new_coeffs_2, wavelet, 'periodization')
    needle = np.abs(fecg * pulse)
    peak_idx = processing.find_local_peaks(needle, 150)
    return peak_idx


def find_onset_offset(fecg, peak_idx):
    """find the positions of QRS onset, peak and offset points"""

    fecg_smooth = savgol_filter(fecg, 21, 3)
    # plt.plot(fecg_smooth, 'r-')
    nbeat = len(peak_idx)
    print("Number of beats:", nbeat)

    onset_idx = []
    offset_idx = []
    pop_list = []
    for i in range(nbeat):
        # if fecg[peak_idx[i]] > 0:
        #
        #     j = 1
        #     while peak_idx[i] - j >= 0 and fecg[peak_idx[i] - j] <= fecg[peak_idx[i] - j + 1]:
        #         j += 1
        #     while peak_idx[i] - j >= 0 and fecg[peak_idx[i] - j] > fecg[peak_idx[i] - j + 1]:
        #         j += 1
        #     if peak_idx[i] - j >= 0:
        #         onset_idx.append(peak_idx[i] - j)
        #     else:
        #         pop_list.append(i)
        #
        #     j = 1
        #     while peak_idx[i] + j < len(fecg) and fecg[peak_idx[i] + j - 1] >= fecg[peak_idx[i] + j]:
        #         j += 1
        #     while peak_idx[i] + j < len(fecg) and fecg[peak_idx[i] + j - 1] < fecg[peak_idx[i] + j]:
        #         j += 1
        #     if peak_idx[i] + j < len(fecg):
        #         offset_idx.append(peak_idx[i] + j)
        #     else:
        #         pop_list.append(i)
        #
        # else:
        #
        #     j = 1
        #     while peak_idx[i] - j >= 0 and fecg[peak_idx[i] - j] >= fecg[peak_idx[i] - j + 1]:
        #         j += 1
        #     while peak_idx[i] - j >= 0 and fecg[peak_idx[i] - j] < fecg[peak_idx[i] - j + 1]:
        #         j += 1
        #     if peak_idx[i] - j >= 0:
        #         onset_idx.append(peak_idx[i] - j)
        #     else:
        #         pop_list.append(i)
        #
        #     j = 1
        #     while peak_idx[i] + j < len(fecg) and fecg[peak_idx[i] + j - 1] <= fecg[peak_idx[i] + j]:
        #         j += 1
        #     while peak_idx[i] + j < len(fecg) and fecg[peak_idx[i] + j - 1] > fecg[peak_idx[i] + j]:
        #         j += 1
        #     if peak_idx[i] + j < len(fecg):
        #         offset_idx.append(peak_idx[i] + j)
        #     else:
        #         pop_list.append(i)

        if fecg_smooth[peak_idx[i]] > 0:

            j = 1
            while peak_idx[i] - j >= 0 and fecg_smooth[peak_idx[i] - j] <= fecg_smooth[peak_idx[i] - j + 1]:
                j += 1
            while peak_idx[i] - j >= 0 and fecg_smooth[peak_idx[i] - j] > fecg_smooth[peak_idx[i] - j + 1]:
                j += 1
            if peak_idx[i] - j >= 0:
                onset_idx.append(peak_idx[i] - j)
            else:
                pop_list.append(i)

            j = 1
            while peak_idx[i] + j < len(fecg_smooth) and \
                    fecg_smooth[peak_idx[i] + j - 1] >= fecg_smooth[peak_idx[i] + j]:
                j += 1
            while peak_idx[i] + j < len(fecg_smooth) and \
                    fecg_smooth[peak_idx[i] + j - 1] < fecg_smooth[peak_idx[i] + j]:
                j += 1
            if peak_idx[i] + j < len(fecg_smooth):
                offset_idx.append(peak_idx[i] + j)
            else:
                pop_list.append(i)

        else:

            j = 1
            while peak_idx[i] - j >= 0 and fecg_smooth[peak_idx[i] - j] >= fecg_smooth[peak_idx[i] - j + 1]:
                j += 1
            while peak_idx[i] - j >= 0 and fecg_smooth[peak_idx[i] - j] < fecg_smooth[peak_idx[i] - j + 1]:
                j += 1
            if peak_idx[i] - j >= 0:
                onset_idx.append(peak_idx[i] - j)
            else:
                pop_list.append(i)

            j = 1
            while peak_idx[i] + j < len(fecg_smooth) and \
                    fecg_smooth[peak_idx[i] + j - 1] <= fecg_smooth[peak_idx[i] + j]:
                j += 1
            while peak_idx[i] + j < len(fecg_smooth) and \
                    fecg_smooth[peak_idx[i] + j - 1] > fecg_smooth[peak_idx[i] + j]:
                j += 1
            if peak_idx[i] + j < len(fecg_smooth):
                offset_idx.append(peak_idx[i] + j)
            else:
                pop_list.append(i)
    print("Pop idx:", pop_list)
    if pop_list:
        if len(pop_list) % 2 == 0:
            peak_idx = np.delete(peak_idx, pop_list)
            onset_idx.pop(-1)
            offset_idx.pop(0)
        elif len(pop_list) % 2 == 1:
            peak_idx = np.delete(peak_idx, pop_list)
            offset_idx.pop(0)
    plt.plot(peak_idx, fecg[peak_idx], 'rx', marker='x', color='#8b0000', label='R_Peak', markersize=8)
    plt.plot(onset_idx, fecg[onset_idx], 'rx', marker='o', color='#8b0000', label='Onset', markersize=8)
    plt.plot(offset_idx, fecg[offset_idx], 'rx', marker='v', color='#8b0000', label='Offset', markersize=8)
    print("R peaks locations:", peak_idx, len(peak_idx))
    print("Onset locations:", onset_idx, len(onset_idx))
    print("Offset locations:", offset_idx, len(offset_idx))
    return onset_idx, peak_idx, offset_idx


def get_reference_voltage(fecg, peak_idx, onset_idx, offset_idx):
    nbeat = len(peak_idx)
    mean_idx_diff_1 = int(np.sum(peak_idx - onset_idx) / nbeat)
    mean_idx_diff_2 = int(np.sum(-peak_idx + offset_idx) / nbeat)
    reference_voltage = np.sum(fecg[onset_idx]) / nbeat
    print("Reference voltage:", reference_voltage)
    return mean_idx_diff_1, mean_idx_diff_2, reference_voltage


def find_t_peak(signal, R_R_interval, offset_located):
    """find the positions of T wave peaks"""

    countT = np.zeros(1, dtype=np.int32)
    pop_list = []
    for i in range(offset_located.shape[0]):
        offset_T_interval = []
        try:
            idx = processing.find_local_peaks(signal[offset_located[i]:offset_located[i] + int(R_R_interval / 2)],
                                              int(R_R_interval / 2) - 10)
            offset_T_interval.append(idx[0])
            idx = idx[0] + offset_located[i]
            countT = np.concatenate((countT, np.zeros(idx - countT.shape[0]), -1 * np.ones(1)))
        except:
            pop_list.append(i)
            average_interval = np.mean(np.array(offset_T_interval)) if offset_T_interval else 0
            if average_interval != 0:
                countT = np.concatenate((countT, np.zeros(offset_located[i] + average_interval
                                                          - countT.shape[0]), -1 * np.ones(1)))
                print("      Add a T peak at {}".format(offset_located[i] + average_interval))
            else:
                print("      No T peak at {}".format(offset_located[i]))
    T_located = np.nonzero(countT)[0]

    return T_located, pop_list


def find_p_peak(signal, R_R_interval, onset_located):
    """find the positions of P wave peaks"""

    countP = np.zeros(1, dtype=np.int32)
    pop_list = []
    for i in range(onset_located.shape[0]):
        onset_P_interval = []
        try:
            idx = processing.find_local_peaks(signal[onset_located[i] - int(R_R_interval / 5):onset_located[i]],
                                              int(R_R_interval / 5) - 1)
            onset_P_interval.append(idx[0])
            idx = idx[0] + onset_located[i] - int(R_R_interval / 5)
            countP = np.concatenate((countP, np.zeros(idx - countP.shape[0]), -1 * np.ones(1)))
        except:
            pop_list.append(i)
            average_interval = np.mean(np.array(onset_P_interval)) if onset_P_interval else 0
            if average_interval != 0:
                countP = np.concatenate((countP, np.zeros(onset_located[i] - int(R_R_interval / 5) +
                                                          average_interval - countP.shape[0]),
                                         -1 * np.ones(1)))
                print("      Add a P peak at {}".format(onset_located[i] - int(R_R_interval / 5) +
                                                        average_interval))
            else:
                print("      No P peak at {}".format(onset_located[i]))
    P_located = np.nonzero(countP)[0]

    return P_located, pop_list


def find_f_point(signal, offset_located, T_located, reference_voltage):
    """find the positions of F points"""

    countF = np.zeros(1, dtype=np.int32)
    pop_list = []
    c_F = 0
    for i in range(offset_located.shape[0]):
        try:
            if np.min(signal[offset_located[i]:T_located[i]]) < reference_voltage < \
                    np.max(signal[offset_located[i]:T_located[i]]):

                idx, = np.where(signal[offset_located[i]:T_located[i]] >= reference_voltage)
                idx = idx[0] + offset_located[i]
                countF = np.concatenate((countF, np.zeros(idx - countF.shape[0]), -1 * np.ones(1)))

            else:
                idx = int((offset_located[i] + T_located[i]) / 2)
                countF = np.concatenate((countF, np.zeros(idx - countF.shape[0]), np.ones(1)))
        except:
            c_F += 1
            pop_list.append(i)
    print("      Pop {} points from total {} points!".format(c_F, offset_located.shape[0]))
    F_located = np.nonzero(countF)[0]
    F_located_val = countF[F_located]

    return F_located, F_located_val, pop_list


def generate_features(fecg, peak_idx, t_peak_idx, f_point_idx, f_point_val,
                      mean_idx_diff_1, mean_idx_diff_2, reference_voltage):
    """extract some features"""
    """features from paper Ischemia episode detection......"""
    nbeat = len(peak_idx)

    feature_1 = []
    feature_2 = []
    feature_3 = []
    for i in range(nbeat):
        k = peak_idx[i] + mean_idx_diff_2
        feature_1.append(np.sum(np.abs(fecg[k:t_peak_idx[i] + 1] - reference_voltage)))
        if f_point_val[i] == -1:
            feature_2.append(np.sum(fecg[k:f_point_idx[i] + 1] - reference_voltage) / np.abs(fecg[peak_idx[i]]))
        else:
            feature_2.append(10)
        m = peak_idx[i] - mean_idx_diff_1
        feature_3.append(np.abs((fecg[k] - fecg[m]) / (k - m)))
    feature_2 = np.asarray(feature_2)
    f2_f_idx = np.where(feature_2 == 10)[0]
    f2_normal_idx = np.where(feature_2 != 10)[0]
    average_f2 = np.sum(feature_2[f2_normal_idx]) / f2_normal_idx.shape[0]
    feature_2[f2_f_idx] = average_f2
    feature_2.tolist()

    feature_1 = np.asarray(feature_1).reshape([len(feature_1), 1])
    feature_2 = np.asarray(feature_2).reshape([len(feature_2), 1])
    feature_3 = np.asarray(feature_3).reshape([len(feature_3), 1])
    return feature_1, feature_2, feature_3

    # x_1 = []
    # x_2 = []
    # x_3 = []
    # for i in range(int(nbeat / 5)):
    #     x_1.append(sum(feature_1[5 * i:5 * i + 5]) / 5)
    #     x_2.append(sum(feature_2[5 * i:5 * i + 5]) / 5)
    #     x_3.append(sum(feature_3[5 * i:5 * i + 5]) / 5)
    # if nbeat % 5 != 0:
    #     x_1.append(sum(feature_1[int(nbeat / 5) * 5 - len(feature_1):]) / (len(feature_1) - int(nbeat / 5) * 5))
    #     x_2.append(sum(feature_2[int(nbeat / 5) * 5 - len(feature_2):]) / (len(feature_2) - int(nbeat / 5) * 5))
    #     x_3.append(sum(feature_3[int(nbeat / 5) * 5 - len(feature_3):]) / (len(feature_3) - int(nbeat / 5) * 5))
    # x_1 = np.asarray(x_1).reshape([len(x_1), 1])
    # x_2 = np.asarray(x_2).reshape([len(x_2), 1])
    # x_3 = np.asarray(x_3).reshape([len(x_3), 1])
    #
    # return x_1, x_2, x_3


atr_list = generate_atr_list('./edb_records')
with open('./edb_st_episodes.json') as infile:
    st_dict = json.load(infile)
