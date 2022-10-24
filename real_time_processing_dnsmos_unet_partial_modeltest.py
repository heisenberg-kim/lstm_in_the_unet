"""
This is an example how to implement real time processing of the DTLN tf light
model in python.

Please change the name of the .wav file at line 43 before running the sript.
For .whl files of the tf light runtime go to:
    https://www.tensorflow.org/lite/guide/python

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 30.06.2020

This code is licensed under the terms of the MIT-license.
"""

"""
this code is modification of DTLN codes(Nils L. Westhausen)
to inference with quantized model(lstm_in_the_unet).

modified by : Kwangyoung Kim(kky0757@gmail.com)

This code is licensed under the terms of the MIT-license.
"""


import soundfile as sf
import numpy as np
#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import time
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


##########################
#set parameters for real time processing(stft, overlap)
block_len = 512
block_shift = 256

#set paths
models_dir = 'path to models'
dirname = 'path to test noisy files'
eval_path = 'path to evaluated files'
#get name(model2 only) of models in a directory  
list_models_dir = os.listdir(models_dir)
models = [model for model in list_models_dir if model.endswith('2.tflite')]

for i, model in enumerate(models):

    print(model)
    print("{}/{} model-testing".format(i+1, len(models)))
    model_name = model.split('/')[-1]
    # [:-7] means ".tflite"
    eval_dir = eval_path + '/' + str(model_name[:-7])
    if os.path.isdir(eval_dir) == False:
        os.mkdir(eval_dir)
    audio_files = os.listdir(dirname)
    for i in range(len(audio_files)):

        print("{}/{}".format(i+1, len(audio_files)))
        audio_dir = os.path.join(dirname, audio_files[i])
        audio, fs = sf.read(audio_dir)
        
        #set & get interpreter for model1
        interpreter_1 = tflite.Interpreter(model_path=models_dir + '/' + model[:-8] + '1.tflite')
        interpreter_1.allocate_tensors()
        input_details_1 = interpreter_1.get_input_details()
        output_details_1 = interpreter_1.get_output_details()

        #set & get interpreter for model2
        interpreter_2 = tflite.Interpreter(model_path= models_dir + '/' + model)
        interpreter_2.allocate_tensors()
        input_details_2 = interpreter_2.get_input_details()
        output_details_2 = interpreter_2.get_output_details()

        #set & get interpreter for model3
        interpreter_3 = tflite.Interpreter(model_path= models_dir + '/' + model[:-8] + '3.tflite')
        interpreter_3.allocate_tensors()
        input_details_3 = interpreter_3.get_input_details()
        output_details_3 = interpreter_3.get_output_details()

        # create states for the lstms
        states_1 = np.zeros(input_details_2[1]['shape']).astype('float32')
        
        # load audio file at 16k fs (please change)
        # check for sampling rate
        if fs != 16000:
            raise ValueError('This model only supports 16k sampling rate.')
        # preallocate output audio
        out_file = np.zeros((len(audio)))
        # create buffer
        in_buffer = np.zeros((block_len)).astype('float32')
        out_buffer = np.zeros((block_len)).astype('float32')
        # calculate number of blocks
        num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
        time_array = []
        
        # iterate over the number of blcoks
        for idx in range(num_blocks):
            # shift values and write to buffer
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]

            # calculate fft of input block
            in_block_fft = np.fft.rfft(np.hamming(block_len) * in_buffer)
            in_mag = np.abs(in_block_fft)
            in_mag_original = in_mag

            in_phase = np.angle(in_block_fft)

            # set tensors to the first model
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')

            interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)

            # run calculation
            interpreter_1.invoke()
            
            conv3_out = interpreter_1.get_tensor(output_details_1[0]['index'])
            pool1 = interpreter_1.get_tensor(output_details_1[1]['index'])
            pool2 = interpreter_1.get_tensor(output_details_1[2]['index'])


            input_scale_conv3_out, input_zero_point_conv3_out = input_details_2[0]["quantization"]
            input_scale_states_1, input_zero_point_states_1 = input_details_2[1]["quantization"]

            conv3_out = conv3_out / input_scale_conv3_out + input_zero_point_conv3_out
            states_1 = states_1 / input_scale_states_1 + input_zero_point_states_1
            
            conv3_out = np.reshape(np.round(conv3_out, 0), (1, 1, -1)).astype('int8')
            states_1 = np.reshape(np.round(states_1, 0), input_details_2[1]['shape']).astype('int8')

            interpreter_2.set_tensor(input_details_2[0]['index'], conv3_out)
            interpreter_2.set_tensor(input_details_2[1]['index'], states_1)

            interpreter_2.invoke()

            flstm_out = interpreter_2.get_tensor(output_details_2[0]['index'])
            states_1 = interpreter_2.get_tensor(output_details_2[1]['index'])

            output_scale_flstm_out, output_zero_point_flstm_out = output_details_2[0]["quantization"]
            output_scale_states_1, output_zero_point_states_1 = output_details_2[1]["quantization"]
            flstm_out = (flstm_out + 128) / 256
            states_1 = (states_1 - output_zero_point_states_1) * output_scale_states_1

            flstm_out = np.reshape(flstm_out, (1, 1, -1)).astype('float32')
            states_1 = np.reshape(states_1, input_details_2[1]['shape']).astype('float32')

            interpreter_3.set_tensor(input_details_3[0]['index'], flstm_out)
            interpreter_3.set_tensor(input_details_3[1]['index'], pool1)
            interpreter_3.set_tensor(input_details_3[2]['index'], pool2)

            interpreter_3.invoke()

            out_mask = interpreter_3.get_tensor(output_details_3[0]['index'])
            estimated_complex = in_mag_original * out_mask * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex)

            # reshape the time domain block
            estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')

            # shift values and write to buffer
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))

            out_buffer += np.squeeze(estimated_block)

            # write block to output file
            out_file[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

        # write to .wav file considering delay(256 shift)
        #out_file[:-256] = out_file[256:]
        
        # write to .wav file
        sf.write(os.path.join(eval_dir, audio_files[i]), out_file, fs)
