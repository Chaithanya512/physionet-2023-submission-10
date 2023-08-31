#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
from sklearn.impute import SimpleImputer
import joblib
import mne
import pywt
from scipy.signal import detrend
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    metadata_features = list() # Array which only contains metadata of the patients.
    outcomes = list()
    cpcs = list()
    
    eegdata_features = list() # Array which only contain eeg features, if atleast one EEG file exists.
    
    

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))
            
        
        current_metadata = get_metadata(data_folder,patient_ids[i])
        patient_metadata = load_challenge_data(data_folder,patient_ids[i])
        outcome = get_outcome(patient_metadata)
        cpc = get_cpc(patient_metadata)
        
        best_eeg_data, sampling_frequency = get_eeg_data(data_folder, patient_ids[i])
        
        if best_eeg_data is not None:
            eeg_features = extract_eeg_features(best_eeg_data, sampling_frequency)
            
        else:
            eeg_features = float("nan")*np.ones(2688)
        
            
            
        metadata_features.append(current_metadata)
        outcomes.append(outcome)
        cpcs.append(cpc)
        
        
        eegdata_features.append(eeg_features)
    

    metadata_features = np.vstack(metadata_features)  # dataframe which only contains metadata of patients.
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    
    eegdata_features = np.vstack(eegdata_features)
    finaldata_features = np.hstack((metadata_features, eegdata_features)) #Add this line to combine the EEG patient's metadata and their EEG features.
    

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')
        

    
    param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [300, 350],
    'base_score':[0.3, 0.4, 0.5, 0.6]
    }
    
        
    
    ###########################################################################################################
    
    
    # Impute any missing features; use the mean value by default.
    
    imputer = SimpleImputer().fit(finaldata_features)
    finaldata_features = imputer.transform(finaldata_features)
    
    
    pca = PCA(n_components=122)
    finaldata_features = pca.fit_transform(finaldata_features)
    
    # Create XGBoost classifier and regressor
    xgb_classifier = xgb.XGBClassifier()
    xgb_regressor = xgb.XGBRegressor()
    
        
    grid_search_clf = GridSearchCV(xgb_classifier, param_grid=param_grid, cv=5)
    grid_search_clf.fit(finaldata_features, outcomes.ravel())
    best_rf_gb_clf = grid_search_clf.best_estimator_

    
    # Train the XGBoost with GridSearchCV for hyperparameter tuning.
    grid_search_reg = GridSearchCV(xgb_regressor, param_grid=param_grid, cv=5)
    grid_search_reg.fit(finaldata_features, cpcs.ravel())
    best_rf_gb_reg = grid_search_reg.best_estimator_
    
    
    # Fit the best parameters to the models.
    outcome_model = best_rf_gb_clf.fit(finaldata_features, outcomes.ravel())
    cpc_model = best_rf_gb_reg.fit(finaldata_features, cpcs.ravel())
    
    
    #Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, pca)
    

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    
    
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    pca = models['pca']
    
    metadata = get_metadata(data_folder, patient_id)
    
    best_eeg_data, sampling_frequency = get_eeg_data(data_folder, patient_id)
    
    if best_eeg_data is not None:
        eeg_features = extract_eeg_features(best_eeg_data, sampling_frequency)
    else:
        eeg_features = float("nan")*np.ones(2688)
        
        
    eeg_features = eeg_features.flatten()
    finaldata_features = np.hstack((metadata, eeg_features))
    finaldata_features = finaldata_features.reshape(1,-1)
    finaldata_features = imputer.transform(finaldata_features)
    finaldata_features = pca.transform(finaldata_features)
    
    outcome = outcome_model.predict(finaldata_features)[0]
    outcome_probability = outcome_model.predict_proba(finaldata_features)[0, 1]
    
    cpc = cpc_model.predict(finaldata_features)[0]
        

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, pca):
    d = { 'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model, 'pca': pca}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
    
    

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_list):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list

    def forward(self, x):
        num_channels = x.size(1)
        outputs = []

        for num_pool_regions in self.pool_list:
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    pool_height = max(x.size(2) // num_pool_regions, 1)
                    pool_width = max(x.size(3) // num_pool_regions, 1)
                    
                    x1 = ix * pool_width
                    x2 = ix * pool_width + pool_width
                    y1 = jy * pool_height
                    y2 = jy * pool_height + pool_height

                    x_crop = x[:, :, y1:y2, x1:x2]
                    pooled_val = F.max_pool2d(x_crop, kernel_size=(pool_height, pool_width))
                    outputs.append(pooled_val)

        outputs = torch.cat(outputs, 1)
        return outputs


    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        
        self.spp = SpatialPyramidPooling(pool_list=[1, 2, 4])
        
    # x represents our data
    def forward(self, x):
        
        x = x.unsqueeze(1)
        
        x = x.float()
        
        # Pass data through conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Pass data through conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Pass data through conv3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        
        x = self.spp(x)
        x = x.view(x.size(0), -1)
        
        return x
        

# Preprocess data.
def preprocess_eeg_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]
    

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=-1, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=-1, verbose='error')
    

    resampling_frequency = 100
        
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)
    

    return data, resampling_frequency


# Reorder/reselect the channels.
def reorder_recording_channels(current_data, current_channels, reordered_channels):
    
    
    if current_channels == reordered_channels:
        return current_data
    else:
        indices = list()
        for channel in reordered_channels:
            if channel in current_channels:
                i = current_channels.index(channel)
                indices.append(i)
        num_channels = len(reordered_channels)
        num_samples = np.shape(current_data)[1]
        reordered_data = np.zeros((num_channels, num_samples))
        reordered_data[:, :] = current_data[indices, :]
    
        return reordered_data
    

def get_metadata(data_folder,patient_id):
    
    patient_metadata = load_challenge_data(data_folder,patient_id)
    patient_features = get_patient_features(patient_metadata)
    
    return patient_features


# Original artifact detection code.


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
   
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]


def peak_detect(signal, max_change_points, min_change_amp, lookahead=200, delta=0):
    # signal: #channel x #points
    res = []
    for cid in range(signal.shape[0]):
        local_max, local_min = peakdetect(signal[cid], lookahead=lookahead, delta=delta)
        if len(local_min)<=0 and len(local_max)<=0:
            res.append(False)
        else:
            if len(local_min)<=0:
                local_extremes = np.array(local_max)
            elif len(local_max)<=0:
                local_extremes = np.array(local_min)
            else:
                local_extremes = np.r_[local_max, local_min]
            local_extremes = local_extremes[np.argsort(local_extremes[:,0])]
            res.append(np.logical_and(np.diff(local_extremes[:,0])<=max_change_points, np.abs(np.diff(local_extremes[:,1]))>=min_change_amp).sum())
    return res


def segment_EEG(EEG, window_time, step_time, Fs, amplitude_thres, n_jobs, to_remove_mean):
    
    std_thres1 = 0.2
    std_thres2 = 0.5
    flat_seconds = 2
    bandpass_freq = [0.5, 30.0]
    num_artifacts = 0

    ## KEEP AN EYE ON IT
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=1, keepdims=True)
        
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    
    ## start_ids
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    
        
    if len(start_ids) <= 0:
        raise ValueError('No EEG segments')
    
    EEG_segs = EEG[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    
    
     ## find large amplitude in signal
    amplitude_large2d = np.max(EEG_segs,axis=2)-np.min(EEG_segs,axis=2)>2*amplitude_thres
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    num_artifacts = num_artifacts + len(amplitude_large1d)
    
            
    ## find flat signal
    # careful about burst suppression
    EEG_segs_temp = EEG_segs[:,:,:(EEG_segs.shape[2]//flat_length)*flat_length]
    short_segs = EEG_segs_temp.reshape(EEG_segs_temp.shape[0], EEG_segs_temp.shape[1], EEG_segs_temp.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres1, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    
    num_artifacts = num_artifacts + len(flat1d)
    
    
#     ## find overly fast rising/decreasing signal
    
#     max_change_points = 0.1*Fs
#     min_change_amp = 1.8*amplitude_thres
#     fast_rising2d = Parallel(n_jobs=n_jobs, verbose=True)(delayed(peak_detect)(EEG_segs[sid], max_change_points, min_change_amp, lookahead=50, delta=0) for sid in range(EEG_segs.shape[0]))
#     fast_rising2d = np.array(fast_rising2d)>0
#     fast_rising1d = np.where(np.any(fast_rising2d, axis=1))[0]
    
#     print("overly fast rising/decreasing signal artifacts: ", len(fast_rising1d))
    
#     num_artifacts = num_artifacts + len(fast_rising1d)
    
    
        
    ## calculate spectrogram
    
    BW = 2.
    specs, freq = mne.time_frequency.psd_array_multitaper(EEG_segs, Fs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], adaptive=False, low_bias=False, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    df = freq[1]-freq[0]
    specs = 10*np.log10(specs.transpose(0,2,1))
    
    
    ## find nan in spectrum
    
    specs[np.isinf(specs)] = np.nan
    nan2d = np.any(np.isnan(specs), axis=1)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    
    nonan_spec_id = np.where(np.all(np.logical_not(np.isnan(specs)), axis=(1,2)))[0]
    
    if len(nonan_spec_id) > 0:
        
        ## find staircase-like spectrum
        # | \      +-+
        # |  \     | |
        # |   -----+ +--\
        # +--------------=====
        
        spec_smooth_window = int(round(1./df))  # 1 Hz
        specs2 = specs[nonan_spec_id][:,np.logical_and(freq>=5,freq<=20)]
        freq2 = freq[np.logical_and(freq>=5,freq<=20)][spec_smooth_window:-spec_smooth_window]
        ww = np.hanning(spec_smooth_window*2+1)
        ww = ww/ww.sum()
        
        smooth_specs = np.apply_along_axis(lambda m: np.convolve(m, ww, mode='valid'), axis=1, arr=specs2)
        dspecs = specs2[:,spec_smooth_window:-spec_smooth_window]-smooth_specs
        dspecs = dspecs-dspecs.mean(axis=1,keepdims=True)
        
        aa = np.apply_along_axis(lambda m: np.convolve(m, np.array([-1.,-1.,0,1.,1.,1.,1.]), mode='same'), axis=1, arr=dspecs)  # increasing staircase-like pattern
        
        bb = np.apply_along_axis(lambda m: np.convolve(m, np.array([1.,1.,1.,1.,0.,-1.,-1.]), mode='same'), axis=1, arr=dspecs)  # decreasing staircase-like pattern
        
        stsp2d = np.logical_or(np.maximum(aa,bb).max(axis=1)>=10, np.any(np.abs(np.diff(specs2,axis=1))>=11, axis=1))
        stsp1d = nonan_spec_id[np.any(stsp2d, axis=1)]  
        
        
        num_artifacts = num_artifacts + len(stsp1d)
               
    return num_artifacts



def get_eeg_data(data_folder, patient_id):
    
    recording_ids = find_recording_files(data_folder, patient_id)
    eeg_group = 'EEG'
    
    window_time = int(5*60) # 5 minutes
    step_time = int(5*60) # 5 minutes
    
    eeg_data_exists = any(
        os.path.exists(os.path.join(data_folder, patient_id, '{}_{}.hea'.format(recording_id, eeg_group)))
        for recording_id in recording_ids
    )
    
    eeg_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'Fp2', 'F8', 'T4', 'T6', 'O2', 'F3', 'C3', 'P3', 'F4', 'C4', 'P4', 'Fz', 'Cz', 'Pz']
    
    
# Bipolar channels: ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    
    if eeg_data_exists:
        
        best_eeg_data = list()
        
        for recording_id in recording_ids:
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, eeg_group))
            
            if os.path.exists(recording_location + '.hea'):
                signal_qualities = []
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data = reorder_recording_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_eeg_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[1, :], data[1, :] - data[2, :], data[2, :]- data[3, :], data[3, :] - data[4, :], data[5, :] - data[6, :], data[6, :] - data[7, :], data[7, :] - data[8, :], data[8, :] - data[9, :], data[0, :] - data[10, :], data[10, :] - data[11, :], data[11, :] - data[12, :], data[12, :] - data[4, :], data[5, :] - data[13, :], data[13, :] - data[14, :], data[14, :] - data[15, :], data[15, :] - data[9, :], data[16, :] - data[17, :], data[17, :] - data[18, :]]) # Convert to bipolar montage
                
                sampling_frequency = int(sampling_frequency)
                fivemin_check = data.shape[1] // (window_time*sampling_frequency)
                
                if fivemin_check > 0: # condition to check if a signal has min 5 minutes of data
                    
                    for start_idx in range(0, data.shape[1]-window_time*sampling_frequency + 1, step_time*sampling_frequency):
                        fivemin_segment = data[:, start_idx:start_idx + window_time * sampling_frequency]
                        num_artifacts = segment_EEG(fivemin_segment, 5, 5, sampling_frequency, amplitude_thres=500, n_jobs=-1, to_remove_mean=False)
                        signal_quality = (180 - num_artifacts) / 100
                        
                        signal_qualities.append(signal_quality)
                    
                    best_segment_idx = np.argmax(signal_qualities)
                
                    if signal_qualities[best_segment_idx] == (180/100): # take the best 5min.
                        best_segment = data[:, best_segment_idx*step_time*sampling_frequency: (best_segment_idx+1)*step_time*sampling_frequency]
                        
                        best_segment = np.sum(best_segment, axis=0)
                        
                        best_segment = best_segment / 18
                        
                        best_eeg_data.append(best_segment)
                

        if len(best_eeg_data) == 0:
            
            return None, None
        
        return np.hstack((best_eeg_data)), sampling_frequency
    else:
        return None, None


def extract_eeg_features(data, sampling_frequency):
    
    
    frequencies = np.arange(0.5, 30.5, 0.5)  # frequencies
    frequencies = frequencies / sampling_frequency  # normalize
    scales = pywt.frequency2scale('morl', frequencies)
    
    coefficients, frequencies = pywt.cwt(data, scales, 'morl', sampling_period=1)
    
    tensor = torch.tensor(coefficients).unsqueeze(0)
    
    myNet = ConvNet()
    
    features = myNet(tensor)
    
    eeg_features = features.detach().numpy()
    
    return eeg_features
    
    

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features
