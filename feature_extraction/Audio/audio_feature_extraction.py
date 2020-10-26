import glob
import numpy as np
import pandas as pd
import parselmouth
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import librosa


def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    
    n_pulses = parselmouth.praat.call(pulses, "Get number of points")
    n_periods = parselmouth.praat.call(pulses, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)
    
    max_voiced_period = 0.02  # This is the "longest period" parameter in some of the other queries
    periods = [parselmouth.praat.call(pulses, "Get time from index", i+1) - parselmouth.praat.call(pulses, "Get time from index", i) for i in range(1, n_pulses)]
    degree_of_voice_breaks = sum(period for period in periods if period > max_voiced_period) / sound.duration
    
    meanIntensity=parselmouth.Sound(voiceID).get_intensity()
    
    #after
    
    # min_pitch = parselmouth.praat.call(pitch, "Get minimum",  "Hertz", "Parabolic")
    
    aud_feat=[meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, n_pulses, n_periods, degree_of_voice_breaks, meanIntensity]

    return aud_feat


def get_prosodic_features(file_loc):
    
    unit="Hertz"
    
    filename = file_loc
    sound = parselmouth.Sound(file_loc)
    y, sr = librosa.load(file_loc)
    duration = librosa.get_duration(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)
    #1
    SD_energy = np.std(energy)
    #2
    pitch = call(sound, "To Pitch", 0.0, 75, 300)
    #3
    
    intensity=call(sound, "To Intensity", 75, 0)
    
    maxIntensity = call(intensity, "Get maximum", 0, 0,'Parabolic') #Ask if parabolic ok?
    minIntensity= call(intensity, "Get minimum", 0, 0,'Parabolic')
    
    maxPitch=call(pitch,"Get maximum",0,0,unit,'Parabolic')
    minPitch=call(pitch,"Get minimum",0,0,unit,'Parabolic')
    #4
    #5
    voiced_frames = pitch.count_voiced_frames()
    total_frames = pitch.get_number_of_frames()
    #6
    voiced_to_total_ratio = voiced_frames/total_frames
    #7
    voiced_to_unvoiced_ratio =  voiced_frames / (total_frames - voiced_frames)
    
    
    return [SD_energy, maxIntensity, minIntensity, maxPitch, minPitch, voiced_frames, voiced_to_total_ratio, voiced_to_unvoiced_ratio]




files=os.listdir('../../data/earning_calls_data/')

"""Extract 1st set of acoustic features"""

audio_featDict={}
for file in tqdm(files):
    audio_folder='../../data/earning_calls_data/'+file+'/Audio'

    if os.path.isfile("../../data/audio_features1.pkl"): 
        with open("../../data/audio_features1.pkl", 'rb') as f:
            audio_featDict=pickle.load(f)

        if file in audio_featDict:
            continue
    
    audio_featDict[file]={}
    
    for aud_file in os.listdir(audio_folder):
        audio_path=audio_folder+'/'+aud_file
        print(aud_file[:-4])
        sound = parselmouth.Sound(audio_path)
        audio_feat=measurePitch(sound, 75, 500, "Hertz")
        audio_featDict[file][aud_file[:-4]]=audio_feat
        
        with open("../../data/audio_features1.pkl", 'wb') as f:
            pickle.dump(audio_featDict,f)
        
    print("Earning Call: part-1 of acoustic features extracted!!!")



"""Extract 2nd set of acoustic features"""
audio_featDictMark2={}

files=os.listdir('../../data/earning_calls_data/')

for file in tqdm(files):

    audio_folder='../../data/earning_calls_data/'+file+'/Audio'
    
    if os.path.isfile("../../data/audio_features2.pkl"):
        with open("../../data/audio_features2.pkl", 'rb') as f:
            audio_featDictMark2=pickle.load(f)  

    
    audio_featDictMark2[file]={}
    
    for aud_file in os.listdir(audio_folder):
        audio_path=audio_folder+'/'+aud_file
        
        if aud_file[:-4] in audio_featDictMark2[file] and len(audio_featDictMark2[file][aud_file[:-4]])>0:
            continue
        # try:
        print(aud_file[:-4])
        audio_feat=get_prosodic_features(audio_path)
            
        
        audio_featDictMark2[file][aud_file[:-4]]=audio_feat
        
        with open("../../data/audio_features2.pkl", 'wb') as f:
            pickle.dump(audio_featDictMark2,f)
        
    print("Earning Call: part-2 of acoustic features extracted!!!")

