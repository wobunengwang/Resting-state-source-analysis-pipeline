# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:58:22 2023

@author: daizhongpengMNE
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:33:47 2023

@author: daizhongpengMNE
"""

import numpy as np
import os
import mne



# ... your existing code ...
datapath =[
    r'C:\newdata'
              ]




subjects_to_do = np.arange(0, len(datapath))
for sub in subjects_to_do:
    data_path = datapath[sub]
    result_path = data_path
    
    file_name = 'training_raw'
    raw_list = list()
    events_list = list()
    
    
    # for subfile in range(1, 3):
    for subfile in range(1, 2):
        # Read in the data from the Result path
        path_file = os.path.join(result_path,file_name + 'ica-' + str(subfile) + '.fif') 
        raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True,preload=True)
        ## remove power line
        meg_picks = mne.pick_types(raw.info, meg=True)
        # freqs = (50,100,150)
        # raw = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
        raw_list.append(raw)
        # events_list.append(events)
        
        
    
#%% 
raw = mne.concatenate_raws(raw_list,on_mismatch='ignore')
epochs = mne.make_fixed_length_epochs(raw, duration=5, preload=True)

channels = [
    ['MEG0522', 'MEG0523', 'MEG0912', 'MEG0913'],
    ['MEG0512', 'MEG0513', 'MEG0922', 'MEG0923'],
    ['MEG0532', 'MEG0533', 'MEG0942', 'MEG0943'],
    ['MEG2132', 'MEG2133', 'MEG2142', 'MEG2143'],
    ['MEG1742', 'MEG1743', 'MEG2542', 'MEG2543'],
    ['MEG1932', 'MEG1933', 'MEG2332', 'MEG2333'],
    ['MEG1732', 'MEG1733', 'MEG2512', 'MEG2513'],
    ['MEG1712', 'MEG1713', 'MEG2532', 'MEG2533'],
    ['MEG1922', 'MEG1923', 'MEG2342', 'MEG2343'],
    ['MEG2042', 'MEG2043', 'MEG2032', 'MEG2033'],
    ['MEG2012', 'MEG2013', 'MEG2022', 'MEG2023'],
    ['MEG1832', 'MEG1833', 'MEG2242', 'MEG2243'],
    ['MEG0742', 'MEG0743', 'MEG0732', 'MEG0733'],
    ['MEG0712', 'MEG0713', 'MEG0722', 'MEG0723'],
    ['MEG0632', 'MEG0633', 'MEG1042', 'MEG1043'],
    ['MEG0642', 'MEG0643', 'MEG1032', 'MEG1033'],
    ['MEG0612', 'MEG0613', 'MEG1022', 'MEG1023'],
    ['MEG0312', 'MEG0313', 'MEG1212', 'MEG1213'],
    ['MEG0542', 'MEG0543', 'MEG0932', 'MEG0933'],
    ['MEG0122', 'MEG0123', 'MEG1412', 'MEG1413'],
    ['MEG0112', 'MEG0113', 'MEG1422', 'MEG1423'],
    ['MEG0342', 'MEG0343', 'MEG1222', 'MEG1223'],
    ['MEG0332', 'MEG0333', 'MEG1242', 'MEG1243'],
    ['MEG1942', 'MEG1943', 'MEG2322', 'MEG2323'],
    ['MEG1722', 'MEG1723', 'MEG2522', 'MEG2523'],
    ['MEG1532', 'MEG1533', 'MEG2632', 'MEG2633'],
    ['MEG1642', 'MEG1643', 'MEG2432', 'MEG2433'],
    ['MEG1912', 'MEG1913', 'MEG2312', 'MEG2313'],
    ['MEG1632', 'MEG1633', 'MEG2442', 'MEG2443'],
    ['MEG1842', 'MEG1843', 'MEG2232', 'MEG2233'],
    ['MEG1822', 'MEG1823', 'MEG2212', 'MEG2213'],
    ['MEG0432', 'MEG0433', 'MEG1142', 'MEG1143'],
    ['MEG0422', 'MEG0423', 'MEG1112', 'MEG1113'],
    ['MEG0412', 'MEG0413', 'MEG1122', 'MEG1123'],
    ['MEG0222', 'MEG0223', 'MEG1312', 'MEG1313'],
    ['MEG0212', 'MEG0213', 'MEG1322', 'MEG1323'],
    ['MEG0132', 'MEG0133', 'MEG1442', 'MEG1443'],
    ['MEG0142', 'MEG0143', 'MEG1432', 'MEG1433'],
    ['MEG0442', 'MEG0443', 'MEG1132', 'MEG1133'],
    ['MEG0232', 'MEG0233', 'MEG1342', 'MEG1343'],
    ['MEG1542', 'MEG1543', 'MEG2622', 'MEG2623'],
    ['MEG1522', 'MEG1523', 'MEG2642', 'MEG2643'],
    ['MEG1612', 'MEG1613', 'MEG2422', 'MEG2423'],
    ['MEG1812', 'MEG1813', 'MEG2222', 'MEG2223'],
    ['MEG1622', 'MEG1623', 'MEG2412', 'MEG2413'],
    ['MEG0242', 'MEG0243', 'MEG1332', 'MEG1333'],
    ['MEG1512', 'MEG1513', 'MEG2612', 'MEG2613'],
]

values_grad = []  # Create an empty list to store the dictionaries with MEGXXXX and value results

for i, ch_group in enumerate(channels):
    # Use the first MEGXXXX to calculate psds_left
    selection_left = ch_group[0:1]
    epo_spectrum_left = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_left)
    psds_left, freqs = epo_spectrum_left.get_data(return_freqs=True)

    # Use the third MEGXXXX to calculate psds_right
    selection_right = ch_group[2:3]
    epo_spectrum_right = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_right)
    psds_right, freqs = epo_spectrum_right.get_data(return_freqs=True)

    average_left = np.mean(psds_left)
    average_right = np.mean(psds_right)
    value1 = (average_left - average_right) / (average_left + average_right)
    values_grad.append({"MEG_Left": ch_group[0], "MEG_Right": ch_group[2], "Value": value1})

    # Use the second MEGXXXX to calculate psds_left
    selection_left = ch_group[1:2]
    epo_spectrum_left = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_left)
    psds_left, freqs = epo_spectrum_left.get_data(return_freqs=True)

    # Use the fourth MEGXXXX to calculate psds_right
    selection_right = ch_group[3:4]
    epo_spectrum_right = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_right)
    psds_right, freqs = epo_spectrum_right.get_data(return_freqs=True)

    average_left = np.mean(psds_left)
    average_right = np.mean(psds_right)
    value2 = (average_left - average_right) / (average_left + average_right)
    values_grad.append({"MEG_Left": ch_group[1], "MEG_Right": ch_group[3], "Value": value2})

# Print the list of dictionaries
print(values_grad)

channels=[
['MEG0521', 'MEG0911'],
['MEG0511', 'MEG0921'],
['MEG0531', 'MEG0941'],
['MEG0311', 'MEG1211'],
['MEG0541', 'MEG0931'],
['MEG0611', 'MEG1021'],
['MEG0121', 'MEG1411'],
['MEG0111', 'MEG1421'],
['MEG0341', 'MEG1221'],
['MEG0321', 'MEG1231'],
['MEG0331', 'MEG1241'],
['MEG0641', 'MEG1031'],
['MEG0131', 'MEG1441'],
['MEG0211', 'MEG1321'],
['MEG0141', 'MEG1431'],
['MEG1511', 'MEG2611'],
['MEG0241', 'MEG1331'],
['MEG1541', 'MEG2621'],
['MEG1521', 'MEG2641'],
['MEG1611', 'MEG2421'],
['MEG1621', 'MEG2411'],
['MEG1631', 'MEG2441'],
['MEG1641', 'MEG2431'],
['MEG1841', 'MEG2231'],
['MEG1741', 'MEG2541'],
['MEG2141', 'MEG2131'],
['MEG1931', 'MEG2331'],
['MEG1731', 'MEG2511'],
['MEG1921', 'MEG2341'],
['MEG1911', 'MEG2311'],
['MEG2041', 'MEG2031'],
['MEG1711', 'MEG2531'],
['MEG1941', 'MEG2321'],
['MEG1721', 'MEG2521'],
['MEG1531', 'MEG2631'],
['MEG1811', 'MEG2221'],
['MEG2011', 'MEG2021'],
['MEG1831', 'MEG2241'],
['MEG0741', 'MEG0731'],
['MEG0711', 'MEG0721'],
['MEG0631', 'MEG1041'],
['MEG0421', 'MEG1111'],
['MEG0431', 'MEG1141'],
['MEG0441', 'MEG1131'],
['MEG0411', 'MEG1121'],
['MEG0221', 'MEG1311'],
['MEG0231', 'MEG1341']]

values_mag = []  # Create an empty list to store the dictionaries with MEGXXXX and value results

for i, ch_group in enumerate(channels):
    # Use the first MEGXXXX to calculate psds_left
    selection_left = ch_group[0:1]
    epo_spectrum_left = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_left)
    psds_left, freqs = epo_spectrum_left.get_data(return_freqs=True)

    # Use the third MEGXXXX to calculate psds_right
    selection_right = ch_group[1:2]
    epo_spectrum_right = epochs.compute_psd(method="multitaper", tmin=0, tmax=2.99, fmin=4, fmax=8, picks=selection_right)
    psds_right, freqs = epo_spectrum_right.get_data(return_freqs=True)

    average_left = np.mean(psds_left)
    average_right = np.mean(psds_right)
    value1 = (average_left - average_right) / (average_left + average_right)
    values_mag.append({"MEG_Left": ch_group[0], "MEG_Right": ch_group[1], "Value": value1})

    

# Print the list of dictionaries
print(values_mag)