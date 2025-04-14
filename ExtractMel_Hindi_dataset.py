import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle

eps = 1e-5

# [Keep all the helper functions like wgn, getlogspec, read_file, etc. unchanged until generate_label]
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def getlogspec(signal,samplerate=16000,winlen=0.02,winstep=0.01,
               nfilt=26,nfft=399,lowfreq=0,highfreq=None,preemph=0.97,
               winfunc=lambda x:np.ones((x,))):
    highfreq= highfreq or samplerate/2
    signal = ps.sigproc.preemphasis(signal,preemph)
    frames = ps.sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames,nfft)
    return pspec 

def read_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def zscore(data,mean,std):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        data[i,:,:,0] = (data[i,:,:,0]-mean)/(std)
    return data

def normalization(data):
    '''
    #apply zscore
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    train_data = zscore(train_data,mean,std)
    test_data = zscore(test_data,mean,std)
    '''
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    data = (data-mean)/std
    return data

def mapminmax(data):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        min = np.min(data[i,:,:,0])
        max = np.max(data[i,:,:,0])
        data[i,:,:,0] = (data[i,:,:,0] - min)/((max - min)+eps)
    return data

def generate_label(emotion, classnum):
    label = -1
    if emotion == 'anger':
        label = 0
    elif emotion == 'sad':
        label = 1
    elif emotion == 'happy':
        label = 2
    elif emotion == 'neutral':
        label = 3
    elif emotion == 'fear':
        label = 4
    elif emotion == 'disgust':
        label = 5
    elif emotion == 'surprise':
        label = 6
    elif emotion == 'sarcastic':
        label = 7
    return label

def load_data():
    # You'll need to create this file or modify to calculate your own stats
    try:
        f = open('./zscore40.pkl','rb')
        mean1, std1, mean2, std2, mean3, std3 = pickle.load(f)
        return mean1, std1, mean2, std2, mean3, std3
    except:
        # Return some default values if file doesn't exist
        return 0, 1, 0, 1, 0, 1
        
def read_Hindi_dataset():
    eps = 1e-5
    filter_num = 40
    
    # Initialize data structures
    all_data = []
    all_labels = []
    
    # Statistics for z-score normalization (you may want to pre-compute these)
    mean1, std1, mean2, std2, mean3, std3 = load_data()
    
    rootdir = 'my Dataset'  # Your dataset root
    
    for session in os.listdir(rootdir):
        if session.startswith('session'):
            session_path = os.path.join(rootdir, session)
            
            for emotion in os.listdir(session_path):
                emotion_path = os.path.join(session_path, emotion)
                
                if os.path.isdir(emotion_path):
                    for wavfile in glob.glob(os.path.join(emotion_path, '*.wav')):
                        # Read and process each WAV file
                        data, time, rate = read_file(wavfile)
                        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                        delta1 = ps.delta(mel_spec, 2)
                        delta2 = ps.delta(delta1, 2)
                        
                        time = mel_spec.shape[0]
                        
                        # Handle variable length audio (pad or split)
                        if time <= 300:
                            part = mel_spec
                            delta11 = delta1
                            delta21 = delta2
                            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant')
                            delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant')
                            delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant')
                            
                            # Create feature cube
                            feature_cube = np.zeros((300, filter_num, 3))
                            feature_cube[:,:,0] = (part - mean1) / (std1 + eps)
                            feature_cube[:,:,1] = (delta11 - mean2) / (std2 + eps)
                            feature_cube[:,:,2] = (delta21 - mean3) / (std3 + eps)
                            
                            all_data.append(feature_cube)
                            all_labels.append(generate_label(emotion, 8))
                        else:
                            # For longer audio, take first and last 300 frames
                            for i in [0, -1]:
                                if i == 0:
                                    begin, end = 0, 300
                                else:
                                    begin, end = time-300, time
                                
                                part = mel_spec[begin:end,:]
                                delta11 = delta1[begin:end,:]
                                delta21 = delta2[begin:end,:]
                                
                                feature_cube = np.zeros((300, filter_num, 3))
                                feature_cube[:,:,0] = (part - mean1) / (std1 + eps)
                                feature_cube[:,:,1] = (delta11 - mean2) / (std2 + eps)
                                feature_cube[:,:,2] = (delta21 - mean3) / (std3 + eps)
                                
                                all_data.append(feature_cube)
                                all_labels.append(generate_label(emotion, 8))
    
    # Convert to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # Shuffle data
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    # Split into train/test/valid (adjust ratios as needed)
    total_samples = len(all_data)
    train_end = int(0.7 * total_samples)
    valid_end = int(0.85 * total_samples)
    
    train_data = all_data[:train_end]
    train_labels = all_labels[:train_end]
    
    valid_data = all_data[train_end:valid_end]
    valid_labels = all_labels[train_end:valid_end]
    
    test_data = all_data[valid_end:]
    test_labels = all_labels[valid_end:]
    
    # Save to pickle file
    output = './Hindi_Emotion_Dataset.pkl'
    with open(output, 'wb') as f:
        pickle.dump((train_data, train_labels, valid_data, valid_labels, test_data, test_labels), f)
    
    print("Dataset processed successfully!")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == '__main__':
    read_Hindi_dataset()