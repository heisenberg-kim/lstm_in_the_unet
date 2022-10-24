# -*- coding: utf-8 -*-
"""
This File contains everything to train the DTLN model.

For running the training see "run_training.py".
To run evaluation with the provided pretrained model see "run_evaluation.py".

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 24.06.2020

This code is licensed under the terms of the MIT-license.
"""


import os, fnmatch
# Tensorflow error message remove
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D, BatchNormalization, MaxPooling1D, GRU, UpSampling1D, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np
import matplotlib.pyplot as plt

class audio_generator():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale 
    audio dataset. This audio generator only supports single channel audio files.
    '''
    
    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        '''
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag=train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object
        self.create_tf_data_obj()
        
    def count_samples(self):
        '''
        Method to list the data of the dataset and count the number of samples. 
        '''

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), '*.wav')
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + \
                int(np.fix(info.data.frame_count/self.len_of_samples))
    
         
    def create_generator(self):
        '''
        Method to create the iterator. 
        '''

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        # iterate over the files  
        for file in self.file_names:
            # read the audio files
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1 or speech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0]/self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                tar_dat = speech[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                # yield the chunks as float32 data
                yield in_dat.astype('float32'), tar_dat.astype('float32')
              

    def create_tf_data_obj(self):
        '''
        Method to to create the tf.data.Dataset. 
        '''

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
                        self.create_generator,
                        (tf.float32, tf.float32),
                        output_shapes=(tf.TensorShape([self.len_of_samples]), \
                                       tf.TensorShape([self.len_of_samples])),
                        args=None
                        )

class audio_generator_2():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale
    audio dataset. This audio generator only supports single channel audio files.
    '''

    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        '''
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag = train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object
        self.create_tf_data_obj()

    def count_samples(self):
        '''
        Method to list the data of the dataset and count the number of samples.
        '''

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), '*.wav')
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + \
                                 int(np.fix(info.data.frame_count / self.len_of_samples))

    def create_generator(self):
        '''
        Method to create the iterator.
        '''

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        # iterate over the files
        for file in self.file_names:
            # read the audio files
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            # speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0] / self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[int(idx * self.len_of_samples):int((idx + 1) *
                                                                  self.len_of_samples)]
                # yield the chunks as float32 data
                yield in_dat.astype('float32')

    def create_tf_data_obj(self):
        '''
        Method to to create the tf.data.Dataset.
        '''

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32),
            output_shapes=(tf.TensorShape([self.len_of_samples])),
            args=None
        )        
                


class DTLN_model():
    '''
    Class to create and train the DTLN model
    '''
    
    def __init__(self):
        '''
        Constructor
        '''

        # defining default cost function
        self.cost_function = self.snr_cost
        # empty property for the model
        self.model = []
        # defining default parameters
        self.fs = 16000
        self.batchsize = 32 
        self.len_samples = 15
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.sumRange = 500
        self.sisdrratio = 5
        self.clipRange = 1.5
        self.compression_constant = 0.3
        self.blockLen = 512
        self.block_shift = 256
        self.dropout = 0.25
        self.take = 10
        self.batch = 1
        self.mag_clip = 10
        self.state_clip = 3
        self.conv3_clip = 2.0
        self.lr = 1e-3
        #self.lr = 5*1e-4
        self.max_epochs = 200
        self.encoder_size = 64
        self.eps = 1e-7
        # reset all seeds to 42 to reduce invariance between training runs
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)





    @staticmethod
    def snr_cost(self, s_estimate, s_true):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''
        
        frames_true = tf.signal.frame(s_true, 512, 512)
        frames_estimate = tf.signal.frame(s_estimate, 512, 512)
        stft_dat_true = tf.signal.rfft(frames_true)
        stft_dat_estimate = tf.signal.rfft(frames_estimate)
        mag_true = tf.abs(stft_dat_true)
        mag_estimate = tf.abs(stft_dat_estimate)
    
        cross = tf.reduce_sum(tf.multiply(s_true, s_estimate), 1, keepdims=True)
        true_norm = tf.reduce_sum(tf.multiply(s_true, s_true), 1, keepdims=True)
        alpha = cross/true_norm
        true_alpha = alpha * s_true
        diff_alpha = true_alpha - s_estimate


        sisdr = tf.reduce_sum(tf.multiply(true_alpha, true_alpha), 1, keepdims=True)/tf.reduce_sum(tf.multiply(diff_alpha, diff_alpha), 1, keepdims=True)
        
        mse = tf.reduce_mean(tf.reduce_mean(mag_true, axis=-1, keepdims=False), axis=-1, keepdims=True) / \
        (tf.reduce_mean(tf.reduce_mean(tf.math.abs(mag_true - mag_estimate), axis=-1, keepdims=False), axis=-1, keepdims=True)+1e-7)
        mse_phase = tf.reduce_mean(tf.reduce_mean(tf.math.abs(stft_dat_true), axis=-1, keepdims=False), axis=-1, keepdims=True) / \
        (tf.reduce_mean(tf.reduce_mean(tf.math.abs(stft_dat_true - stft_dat_estimate), axis=-1, keepdims=False), axis=-1, keepdims=True)+1e-7)
        
        comp_const = self.compression_constant
        comp_complex = tf.complex(tf.constant([self.compression_constant]), tf.constant([self.compression_constant]))
        mse_compressed = tf.reduce_mean(tf.reduce_mean(tf.pow(mag_true, tf.constant(comp_const)), axis=-1, keepdims=False), axis=-1, keepdims=True) / \
        (tf.reduce_mean(tf.reduce_mean(tf.math.abs(tf.pow(mag_true, tf.constant(comp_const)) - tf.pow(mag_estimate, tf.constant(comp_const))), axis=-1, keepdims=False), axis=-1, keepdims=True)+1e-7)
        #for complex number z z^a is not well defined at |z| = 0    
        mse_phase_compressed = (tf.reduce_mean(tf.reduce_mean(tf.math.abs(tf.pow(stft_dat_true + 1e-7,comp_complex)), axis=-1, keepdims=False), axis=-1, keepdims=True)) / \
        (tf.reduce_mean(tf.reduce_mean(tf.math.abs(tf.pow(stft_dat_true + 1e-7,comp_complex) - tf.pow(stft_dat_estimate + 1e-7,comp_complex)), axis=-1, keepdims=False), axis=-1, keepdims=True)+1e-7)
        
        
        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr)
        num_mse = tf.math.log(mse)
        num_sisdr = tf.math.log(sisdr)
        num_mse_phase = tf.math.log(mse_phase)
        num_snr_sisdr = tf.math.log((1-0.1*self.sisdrratio)*snr + 0.1*self.sisdrratio*sisdr)
        num_mse_compressed = tf.math.log(mse_compressed)
        #num_mse_phase_compressed= tf.math.log(mse_phase_compressed)
        num_mse_phase_compressed = tf.math.log(tf.clip_by_value(mse_phase_compressed, 1, 100))
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))
        loss_mse = -10*(num_mse / (denom))
        loss_sisdr = -10*(num_sisdr / (denom))
        loss_mse_phase = -10*(num_mse_phase / (denom))
        loss_snr_sisdr = -10*(num_snr_sisdr / (denom))
        loss_mse_compressed = -10*(num_mse_compressed / (denom))
        loss_mse_phase_compressed = -10*(num_mse_phase_compressed / (denom))

        # returning the loss
        return [loss, loss_mse, loss_sisdr, loss_mse_phase, loss_snr_sisdr, loss_mse_compressed, loss_mse_phase_compressed]
        

    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            # calculating loss and squeezing single dimensions away
            loss_snr = tf.squeeze(self.cost_function(self,y_pred,y_true)[0])
            loss_mse = tf.squeeze(self.cost_function(self,y_pred,y_true)[1])
            loss_sisdr = tf.squeeze(self.cost_function(self,y_pred,y_true)[2])
            loss_mse_phase = tf.squeeze(self.cost_function(self,y_pred,y_true)[3])
            loss_snr_sisdr = tf.squeeze(self.cost_function(self,y_pred,y_true)[4])
            loss_mse_compressed = tf.squeeze(self.cost_function(self,y_pred,y_true)[5])
            loss_mse_phase_compressed = tf.squeeze(self.cost_function(self,y_pred,y_true)[6])

            # calculate mean over batches
            loss_snr = tf.reduce_mean(loss_snr)
            loss_mse = tf.reduce_mean(loss_mse)
            loss_sisdr = tf.reduce_mean(loss_sisdr)
            loss_mse_phase = tf.reduce_mean(loss_mse_phase)
            loss_snr_sisdr = tf.reduce_mean(loss_snr_sisdr)
            loss_mse_compressed = tf.reduce_mean(loss_mse_compressed)
            loss_mse_phase_compressed = tf.reduce_mean(loss_mse_phase_compressed)
            # return the loss
            #return 0.8*loss_snr + 0.1*loss_mse_compressed + 0.1*loss_mse_phase_compressed
            return 0.8*loss_snr + 0.2*loss_mse_compressed
            #return 0.8*loss_snr + 0.1*loss_mse + 0.1*loss_mse_phase
            #return loss_mse_phase_compressed
            #return loss_snr_sisdr + 0.1*loss_mse_compressed
            #return loss_snr
        
        # returning the loss function as handle
        return lossFunction
    
    

    '''
    In the following some helper layers are defined.
    '''  
    
    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        windows = tf.signal.hamming_window(self.blockLen, periodic=True, dtype=tf.dtypes.float32, name=None)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(windows*frames)
        #stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

    def stftLayer2(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''

        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        windows = tf.signal.hamming_window(self.blockLen, periodic=True, dtype=tf.dtypes.float32, name=None)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(windows*frames)
        #stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        #mean_mag = tf.reduce_mean(mag)
        #phase = tf.math.angle(stft_dat)

        # returning magnitude and phase as list
        return mag



    def frontLayers(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        windows = tf.signal.hamming_window(self.blockLen, periodic=True, dtype=tf.dtypes.float32, name=None)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(windows*frames)
        #stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        #mag = tf.clip_by_value(mag, clip_value_min=0, clip_value_max=10)
        conv1 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(mag)
        conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)

        conv2 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(pool1)
        conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(pool2)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(rate=0.25)(conv3)
        
        return conv3
        
    def frontLayers_2(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        windows = tf.signal.hamming_window(self.blockLen, periodic=True, dtype=tf.dtypes.float32, name=None)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(windows*frames)
        #stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        conv1 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(mag)
        conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)

        conv2 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(pool1)
        conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(pool2)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(rate=0.25)(conv3)
        
        flstm = self.seperation_kernel(self.numLayer, 128, conv3)
        return flstm
    def frameLayer(self, x):
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        print(frames.shape)
        return frames

    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer. The layer
        calculates the rFFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # expanding dimensions
        frames = tf.expand_dims(x, axis=1)
        windows = tf.signal.hamming_window(self.blockLen, periodic=True, dtype=tf.dtypes.float32, name=None)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(windows*frames)
        #stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

 
        
    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        
        # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)
    
        

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        '''
        Method to create a separation kernel. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Iynputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense laer
        '''

        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            # using dropout between the LSTM layer for regularization 
            if idx<(num_layer-1):
                x = Dropout(self.dropout)(x)
        norm = BatchNormalization(axis=-1)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(norm)
        mask = Activation(self.activation)(mask)
        # returning the mask
        return mask
   


    def seperation_kernel_with_states(self, num_layer, mask_size, x, 
                                      in_states):
        '''
        Method to create a separation kernel, which returns the LSTM states. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''
        
        states_h = []
        states_c = []
        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            in_state = [in_states[:,idx,:, 0], in_states[:,idx,:, 1]]
            x, h_state, c_state = LSTM(self.numUnits, return_sequences=True, 
                     unroll=True, return_state=True)(x, initial_state=in_state)
            # using dropout between the LSTM layer for regularization 
            if idx<(num_layer-1):
                x = Dropout(self.dropout)(x)
            
            states_h.append(h_state)
            states_c.append(c_state)
        norm = BatchNormalization(axis=-1)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(norm)
        mask = Activation(self.activation)(mask)
        out_states_h = tf.reshape(tf.stack(states_h, axis=0), 
                                  [1,num_layer,self.numUnits])
        out_states_c = tf.reshape(tf.stack(states_c, axis=0), 
                                  [1,num_layer,self.numUnits])
        out_states = tf.stack([out_states_h, out_states_c], axis=-1)
        # returning the mask and states
        return mask, out_states

    def seperation_kernel_with_states2(self, num_layer, mask_size, x,
                                      in_states):
        '''
        Method to create a separation kernel, which returns the LSTM states.
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''

        states_h = []
        states_c = []
        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            in_state = [in_states[:, idx, :, 0], in_states[:, idx, :, 1]]
            x, h_state, c_state = LSTM(self.numUnits, return_sequences=True,
                                       unroll=True, return_state=True)(x, initial_state=in_state)
            # using dropout between the LSTM layer for regularization
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
            states_h.append(h_state)
            states_c.append(c_state)
        norm = BatchNormalization(axis=-1)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(norm)
        mask = Activation(self.activation)(mask)
        out_states_h = tf.reshape(tf.stack(states_h, axis=0),
                                  [1, num_layer, self.numUnits])
        out_states_c = tf.reshape(tf.stack(states_c, axis=0),
                                  [1, num_layer, self.numUnits])
        out_states = tf.stack([out_states_h, out_states_c], axis=-1)
        # returning the mask and states
        return out_states

    def build_DTLN_model(self, norm_stft=False, print_on=1):
        '''
        Method to build and compile the DTLN model. The model takes time domain 
        batches of size (batchsize, len_in_samples) and returns enhanced clips 
        in the same dimensions. As optimizer for the Training process the Adam
        optimizer with a gradient norm clipping of 3 is used. 
        The model contains two separation cores. The first has an STFT signal 
        transformation and the second a learned transformation based on 1D-Conv 
        layer. 
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(None, None))
        mag,angle = Lambda(self.stftLayer)(time_dat)

        conv1 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(mag)
        conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)

        conv2 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(pool1)
        conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(pool2)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(rate=0.25)(conv3)

        flstm = self.seperation_kernel(self.numLayer, 128, conv3)

        up4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(flstm))
        up4 = Activation('relu')(up4)
        concat4 = concatenate([pool2,up4], axis=2)
        conv4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(concat4)
        conv4 = BatchNormalization(axis=-1)(conv4)
        conv4 = Activation('relu')(conv4)

        up5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(conv4))
        up5 = Activation('relu')(up5)
        concat5 = concatenate([pool1, up5], axis=2)
        conv5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(concat5)
        conv5 = BatchNormalization(axis=-1)(conv5)
        conv5 = Activation('relu')(conv5)
        dense = Dense(257)(conv5)
        mask = Activation(self.activation)(dense)

        estimated_mag = Multiply()([mag, mask])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        estimated_sig = Lambda(self.overlapAddLayer)(estimated_frames_1)


        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # show the model summary
        if print_on == 1:
            print(self.model.summary())

    def build_DTLN_model_stateful(self, norm_stft=False):
        '''
        Method to build stateful DTLN model for real time processing. The model 
        takes one time domain frame of size (1, blockLen) and one enhanced frame.
        '''
        time_dat = Input(batch_shape=(1, self.blockLen))
        mag,angle = Lambda(self.fftLayer)(time_dat)
         
        conv1 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(mag)
        conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)

        conv2 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(pool1)
        conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(pool2)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(rate=0.25)(conv3)

        flstm = self.seperation_kernel(self.numLayer, 128, conv3)

        up4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(flstm))
        up4 = Activation('relu')(up4)
        concat4 = concatenate([pool2,up4], axis=2)
        conv4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(concat4)
        conv4 = BatchNormalization(axis=-1)(conv4)
        conv4 = Activation('relu')(conv4)

        up5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(conv4))
        up5 = Activation('relu')(up5)
        concat5 = concatenate([pool1, up5], axis=2)
        conv5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(concat5)
        conv5 = BatchNormalization(axis=-1)(conv5)
        conv5 = Activation('relu')(conv5)
        dense = Dense(257)(conv5)
        mask = Activation(self.activation)(dense)
       
        estimated_mag = Multiply()([mag, mask])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle]) 
        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_frames_1)
        # show the model summary
        print(self.model.summary())

        
    def compile_model(self):
        '''
        Method to compile the model for training

        '''
        
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam)
        
    def create_saved_model(self, weights_file, target_name):
        '''
        Method to create a saved model folder from a weights file

        '''
        # check for type
        if weights_file.find('_norm_') != -1:
            norm_stft = True
        else:
            norm_stft = False
        # build model    
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        # load weights
        self.model.load_weights(weights_file)
        # save model
        tf.saved_model.save(self.model, target_name)
        
    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):
        ''' 
        Method to create a tf lite model folder from a weights file. 
        The conversion creates two models, one for each separation core. 
        Tf lite does not support complex numbers yet. Some processing must be 
        done outside the model.
        For further information and how real time processing can be 
        implemented see "real_time_processing_tf_lite.py".
        
        The conversion only works with TF 2.3.

        '''
        # build model    
        self.build_DTLN_model_stateful()
        # load weights
        self.model.load_weights(weights_file)
         
        #### Model 1 ##########################
        mag = Input(batch_shape=(1, 1, (self.blockLen//2+1)))
        #states_in = Input(batch_shape=(1, self.numLayer, self.numUnits, 2))


        conv1 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(mag)
        conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)
        print(pool1.shape)
        conv2 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(pool1)
        conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(pool2)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(128,1,padding='same',strides=1,use_bias=False)(conv3)
        conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(rate=0.25)(conv3)

        model_1 = Model(inputs=[mag], outputs=[conv3, pool1, pool2])
        #model2
        conv3_out = Input(batch_shape=(1,1,128)) 
        states_in = Input(batch_shape=(1, self.numLayer, self.numUnits, 2))
        
        flstm, states_out = self.seperation_kernel_with_states(self.numLayer, 128, conv3_out, states_in)

        model_2 = Model(inputs=[conv3_out, states_in], outputs=[flstm, states_out])
        
        #model3
        up4_out = Input(batch_shape=(1, 1, 128))
        pool1_out = Input(batch_shape=(1, 1, 32))
        pool2_out = Input(batch_shape=(1, 1, 64))

        print(up4_out.shape)
        up4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(up4_out))
        print(up4.shape)
        up4 = Activation('relu')(up4)
        concat4 = concatenate([pool2_out,up4], axis=2)
        conv4 = Conv1D(64,1,padding='same',strides=1,use_bias=False)(concat4)
        conv4 = BatchNormalization(axis=-1)(conv4)
        conv4 = Activation('relu')(conv4)

        up5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(UpSampling1D(size=1)(conv4))
        up5 = Activation('relu')(up5)
        concat5 = concatenate([pool1_out, up5], axis=2)
        conv5 = Conv1D(32,1,padding='same',strides=1,use_bias=False)(concat5)
        conv5 = BatchNormalization(axis=-1)(conv5)
        conv5 = Activation('relu')(conv5)
        dense = Dense(257)(conv5)
        mask = Activation(self.activation)(dense)

        model_3 = Model(inputs=[up4_out, pool1_out, pool2_out], outputs=[mask])
        # normalizing log magnitude stfts to get more robust against level variations
        
        #model_1 = Model(inputs=[mag, states_in], outputs=[mask, states_out])
        # set weights to submodels
        weights = self.model.get_weights()
        print(len(weights))
        
        weights_clip = weights[20:32]
        
        weight_min = []
        weight_max = []
        weight_mean = []
        weight_std = []
        weight_num = []


        #weights_clip[10] = np.clip(weights[10], -2, 2)
        
        with open("weights.csv", "w") as f:
            f.write("entires" + ',' + "min" + "," + "max" + "," + "mean" + "," + "std" + '\n')
            for i in range(len(weights_clip)):
                globals()['weights_' + str(i)] = np.reshape(weights_clip[i], (-1))

                weight_min.append(np.min(weights_clip[i]))
                weight_max.append(np.max(weights_clip[i]))
                weight_mean.append(np.mean(weights_clip[i]))
                weight_std.append(np.std(weights_clip[i]))
                weight_num.append(np.shape(np.reshape(weights_clip[i], (-1)))[-1])
                f.write(str(weight_num[i]) + ',' + str(weight_min[i]) + ',' + str(weight_max[i]) + ',' + str(weight_mean[i]) + ',' + str(weight_std[i]) + '\n')
        

        #weight clipping
        weights_clip[0] = np.clip(weights_clip[0], -2.5, 1.5)
        weights_clip[1] = np.clip(weights_clip[1], -2, 2)
        weights_clip[3] = np.clip(weights_clip[3], -2, 2)
        weights_clip[4] = np.clip(weights_clip[4], -2, 2)
        weights_clip[10] = np.clip(weights_clip[10], -1, 1)

        #bias clipping

        weights_clip[2] = np.clip(weights_clip[2], -1, 1.5)
        weights_clip[5] = np.clip(weights_clip[5], -1, 1.5)
        weights_clip[6] = np.clip(weights_clip[6], 0.2, 1.0)
        weights_clip[7] = np.clip(weights_clip[7], -0.75, 0.75)
        weights_clip[8] = np.clip(weights_clip[8], -0.3, 0.3)
        weights_clip[9] = np.clip(weights_clip[9], 0, 0.14)
        weights_clip[11] = np.clip(weights_clip[11], -1, 1)
        
        model_1.set_weights(weights[:20])
        model_2.set_weights(weights_clip)
        model_3.set_weights(weights[32:])
        
        plt.figure(figsize=(16, 12))
        plt.subplot(4, 3, 1)
        plt.hist(weights_0, bins=100, log=True)
        
        plt.subplot(4, 3, 2)
        plt.hist(weights_1, bins=100, log=True)
        

        plt.subplot(4, 3, 3)
        plt.hist(weights_2, bins=100, log=True)
        
        plt.subplot(4, 3, 4) 
        plt.hist(weights_3, bins=100, log=True)

        plt.subplot(4, 3, 5) 
        plt.hist(weights_4, bins=100, log=True)

        plt.subplot(4, 3, 6)
        plt.hist(weights_5, bins=100, log=True)

        plt.subplot(4, 3, 7)
        plt.hist(weights_6, bins=100, log=True)
        
        plt.subplot(4, 3, 8)
        plt.hist(weights_7, bins=100, log=True)
        
        plt.subplot(4, 3, 9)
        plt.hist(weights_8, bins=100, log=True)
        
        plt.subplot(4, 3, 10)
        plt.hist(weights_9, bins=100, log=True)
    
        plt.subplot(4, 3, 11)
        plt.hist(weights_10, bins=100, log=True)
       
        plt.subplot(4, 3, 12)
        plt.hist(weights_11, bins=100, log=True)
        
        '''
        plt.subplot(4, 4, 13)
        plt.hist(weights_12, bins=100)
        
        
        plt.subplot(4, 4, 14)
        plt.hist(weights_13, bins=100)
        
        plt.subplot(4, 4, 15)
        plt.hist(weights_14, bins=100)
        
        plt.subplot(4, 4, 16)
        plt.hist(weights_15, bins=100)
        '''
        plt.tight_layout()
        plt.savefig('weights_dirstribution_unet_partial_model2.png')
        #model_1.set_weights(weights)
        #model_1.set_weights(weights_clip)
        plt.clf() 
        
        #model_1.set_weights(weights)
        #model_1.save('model_v3_500h_hamm_lstm_nonstateful_norm.h5')
        #model_2.set_weights(weights[num_elements_first_core:])
        # convert first model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
        
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.post_training_quantize=True
        #converter.representative_dataset = self.representative_data_gen_part1 
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8
        #converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '_1.tflite', 'wb') as f:
              f.write(tflite_model)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
        
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_data_gen_unet_partial 
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '_2.tflite', 'wb') as f:
              f.write(tflite_model)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model_3)
        
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        #converter.representative_dataset = self.representative_data_gen_unet_partial 
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8
        #converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '_3.tflite', 'wb') as f:
              f.write(tflite_model)
        print('TF lite conversion complete!')
        
    def representative_data_gen(self):

        generator_input = audio_generator_2("/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/noisy",\
                                            "/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/clean",
                                            int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift),self.fs)
        dataset = generator_input.tf_data_set

        mag_hist = []
        lstm_state_hist = []
        a = 0
        fc_hist = []
        count=0
        for i in dataset.batch(self.batch).take(self.take):
            #print("length:{}".format(len(dataset.batch(1).take(1))))
            lstm_state = tf.zeros([1, self.numLayer, self.numUnits, 2], tf.float32)
            for j in range(self.stftLayer2(i).shape[1]):
                lstm_state = self.seperation_kernel_with_states2(self.numLayer, (self.blockLen // 2 + 1),
                                                                 tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]),
                                                                 lstm_state)
                
                cliped = tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)
                if tf.reduce_sum(cliped) < self.sumRange:
                    print(tf.reduce_sum(cliped))
                    #yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip), tf.clip_by_value(lstm_state, clip_value_min=-self.state_clip, clip_value_max=self.state_clip)]
                    yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip),lstm_state]
                
    def representative_data_gen_part1(self):

        generator_input = audio_generator_2("/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/noisy",\
                                            "/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/clean",
                                            int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift),self.fs)
        dataset = generator_input.tf_data_set

        mag_hist = []
        lstm_state_hist = []
        a = 0
        fc_hist = []
        count=0
        for i in dataset.batch(self.batch).take(self.take):
            #print("length:{}".format(len(dataset.batch(1).take(1))))
            lstm_state = tf.zeros([1, self.numLayer, self.numUnits, 2], tf.float32)
            for j in range(self.stftLayer2(i).shape[1]):
                lstm_state = self.seperation_kernel_with_states2(self.numLayer, (self.blockLen // 2 + 1),
                                                                 tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]),
                                                                 lstm_state)
                
                cliped = tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)
                if tf.reduce_sum(cliped) < self.sumRange:
                    print(tf.reduce_sum(cliped))
                    #yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip), tf.clip_by_value(lstm_state, clip_value_min=-self.state_clip, clip_value_max=self.state_clip)]
                    yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)]
        
    def representative_data_gen_part3(self):

        generator_input = audio_generator_2("/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/noisy",\
                                            "/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/clean",
                                            int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift),self.fs)
        dataset = generator_input.tf_data_set

        mag_hist = []
        lstm_state_hist = []
        a = 0
        fc_hist = []
        count=0
        for i in dataset.batch(self.batch).take(self.take):
            #print("length:{}".format(len(dataset.batch(1).take(1))))
            lstm_state = tf.zeros([1, self.numLayer, self.numUnits, 2], tf.float32)
            for j in range(self.stftLayer2(i).shape[1]):
                lstm_state = self.seperation_kernel_with_states2(self.numLayer, (self.blockLen // 2 + 1),
                                                                 tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]),
                                                                 lstm_state)
                
                cliped = tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)
                if tf.reduce_sum(cliped) < self.sumRange:
                    print(tf.reduce_sum(cliped))
                    #yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip), tf.clip_by_value(lstm_state, clip_value_min=-self.state_clip, clip_value_max=self.state_clip)]
                    yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip),lstm_state]
    
    def representative_data_gen_unet_partial(self):

        generator_input = audio_generator_2("/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/noisy",\
                                            "/datas/audio/dataset/foreign/DNS-Challenge_2020(breizhn)/synthesis/representative/clean",
                                            int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift),self.fs)
        dataset = generator_input.tf_data_set

        mag_hist = []
        lstm_state_hist = []
        a = 0
        fc_hist = []
        count=0
        output_shape=128 # output shape of kernel, in this case, use 128 = Conv1D(128)
        mask_size = 128
        sum_cliped_list = []
        input_list = []
        states_list = []
        for i in dataset.batch(self.batch).take(self.take):
            #print("length:{}".format(len(dataset.batch(1).take(1))))
            lstm_state = tf.zeros([1, self.numLayer, self.numUnits, 2], tf.float32)
            for j in range(self.frontLayers(i).shape[1]):
                #print(self.frontLayers(i).shape)
                lstm_state = self.seperation_kernel_with_states2(self.numLayer, mask_size,
                                                                 tf.slice(self.frontLayers(i), [0, j, 0], [1, 1, output_shape]),
                                                                 lstm_state)
                input_list.append(tf.slice(self.frontLayers(i), [0, j, 0], [1, 1, 128]))
                states_list.append(np.reshape(lstm_state, (-1)))
                #plt.hist(np.reshape(input_list, (-1)), bins=30)
                #plt.savefig("frontLayer_hist.png")
                '''
                if j == self.frontLayers(i).shape[1]-1:
                    plt.subplot(1,2,1)
                    plt.hist(np.reshape(input_list, (-1)), bins=256, log=True)
                    #plt.savefig("frontLayer_hist.png")
                    plt.subplot(1,2,2)
                    plt.hist(np.reshape(states_list, (-1)), bins=256, log=True)
                    plt.savefig("frontLayer_representative_states_hist.png")
                '''
                
                cliped = tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)
                cliped_reduced_sum = tf.reduce_sum(cliped)
                if cliped_reduced_sum < self.sumRange:
                    print(cliped_reduced_sum)
                    yield [tf.clip_by_value(tf.slice(self.frontLayers(i), [0, j, 0], [1, 1, 128]), clip_value_min=0, clip_value_max=self.conv3_clip), tf.clip_by_value(lstm_state, clip_value_min=-self.state_clip, clip_value_max = self.state_clip)]
                #yield [tf.slice(self.frontLayers(i), [0, j, 0], [1, 1, 128]), lstm_state]
                
                '''
                #print(lstm_state)
                cliped = tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip)
                
                
                sum_cliped_list.append(tf.reduce_sum(cliped))
                
                if count == 300:
                    plt.hist(np.reshape(sum_cliped_list,(-1)), bins=30)
                    plt.savefig("/home/kwkim/emt_dtln/generator/weights_distribution/sum_cliped_list.png")
                count += 1
                if tf.reduce_sum(cliped) < self.sumRange:
                    print(tf.reduce_sum(cliped))
                    yield [tf.slice(self.frontLayers(i), [0, j, 0], [1, 1, 128]), lstm_state]
                    #yield [tf.clip_by_value(tf.slice(Lambda(self.stftLayer2)(i), [0, j, 0], [1, 1, 257]), clip_value_min=0, clip_value_max=self.mag_clip),lstm_state]

                '''
    def train_model(self, model_dir, runName, path_to_train_mix, path_to_train_speech, \
                    path_to_val_mix, path_to_val_speech):
        '''
        Method to train the DTLN model. 
        '''
        
        # create save path if not existent
        savePath = model_dir
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        
        #csv_logger = CSVLogger('/home/kwkim/emt_dtln/generator/gpu1_training_quantization/log_categorized_noise/' + runName + '.log')
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=10, verbose=0, mode='auto', baseline=None)
        # create model check pointer to save the best model
        checkpointer = ModelCheckpoint(savePath+runName+'.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # calculate length of audio chunks in samples
        len_in_samples = int(np.fix(self.fs * self.len_samples / 
                                    self.block_shift)*self.block_shift)
        # create data generator for training data
        generator_input = audio_generator(path_to_train_mix, 
                                          path_to_train_speech, 
                                          len_in_samples, 
                                          self.fs, train_flag=True)
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples//self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(path_to_val_mix,
                                        path_to_val_speech, 
                                        len_in_samples, self.fs)
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of validation steps
        steps_val = generator_val.total_samples//self.batchsize
        # start the training of the model
        self.model.fit(
            x=dataset, 
            batch_size=None,
            steps_per_epoch=steps_train, 
            epochs=self.max_epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=steps_val, 
            callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            max_queue_size=50,
            workers=4,
            use_multiprocessing=True)
        # clear out garbage
        tf.keras.backend.clear_session()

    

class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')
 

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), 
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently 
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs
    

class InstantLayerNormalization_approximation(Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')
 

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean((inputs-mean)*(inputs-mean), axis=[-1], keepdims=True)
        # calculate standard deviation
        std_inv = 3162.28 - 1.58114*1e+10*variance + 1.18585*1e+17*variance*variance
        # normalize each frame independently 
        outputs = (inputs - mean) * std_inv
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs    
