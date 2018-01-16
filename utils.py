import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.platform import gfile
from tensorflow.python.ops import io_ops
import os

def word2label(word, label_words):
    try:
        y = label_words.index(word)
    except ValueError:
        y = -1
    return y

def get_time_shifts(timeshift):
    if timeshift > 0:
        time_shift_amount = np.random.randint(-timeshift, timeshift)
    else:
        time_shift_amount = 0

    if time_shift_amount > 0:
        padding = [[time_shift_amount, 0], [0, 0]]
        offset = [0, 0]
    else:
        padding = [[0, -time_shift_amount], [0, 0]]
        offset = [-time_shift_amount, 0]

    return padding, offset


def get_noise(bg_noise, prob, vol_range, n_samples):
    chosen = np.random.randint(len(bg_noise))
    background_samples = bg_noise[chosen]

    offset = np.random.randint(0, len(background_samples) - n_samples)
    clipped = background_samples[offset:(offset + n_samples)]
    bg = clipped.reshape([n_samples, 1])

    if vol_range == 1.0:
        bg_volume = 1.0
    elif np.random.uniform(0, 1) <= prob:
        bg_volume = np.random.uniform(0, vol_range)
    else:
        bg_volume = 0

    return bg, bg_volume

def get_background_data(path):
    background_data = []
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        search_path = os.path.join(path, '*.wav')
        for wav_path in gfile.Glob(search_path):
            wav_data = sess.run(
                wav_decoder,
                feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
            background_data.append(wav_data)
            print(wav_path, len(wav_data))
    return background_data

def Features(conf):
    sound = tf.placeholder(tf.float32, [None, None], name='wav')
    
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        sound,
        window_size=conf['window_size_samples'],
        stride=conf['window_stride_samples'],
        magnitude_squared=True, name='spectogram'
    )
    
    mfcc = contrib_audio.mfcc(
        spectrogram,
        conf['sample_rate'],
        dct_coefficient_count=conf['dct_coefficient_count'], name='mfcc'
    )
    
    spect_norm = spectrogram/tf.reduce_sum(spectrogram, [1,2])
    mfcc_norm = mfcc/tf.reduce_sum(mfcc, [1,2])
    
    return sound, spectrogram, mfcc, spect_norm, mfcc_norm

def Augmenter(conf):
    desired_samples = conf['desired_samples']
    # Allow the audio sample's volume to be adjusted.
    sample_rate_ = tf.placeholder(tf.int32, [], name='samping-rate')
    wav_ = tf.placeholder(tf.float32, [None, None], name='wav')
    fg_volume_ = tf.placeholder(tf.float32, [], name='fg-volume')
    scaled_fg = tf.multiply(wav_, fg_volume_, name='scaled-fg')

    # Shift the sample's start position, and pad any gaps with zeros.
    padding_ = tf.placeholder(tf.int32, [2, 2], name='padding')
    offset_ = tf.placeholder(tf.int32, [2], name='offset')
    padded_fg = tf.pad(scaled_fg, padding_, mode='CONSTANT', name='padded-fg')
    sliced_fg = tf.slice(padded_fg, offset_, [desired_samples, -1], name='sliced-fg')

    # Mix in background noise.
    bg_ = tf.placeholder(tf.float32, [desired_samples, 1], name='bg')
    bg_volume_ = tf.placeholder(tf.float32, [], name='bg-volume')
    bg_mul = tf.multiply(bg_, bg_volume_, name='bg-mul')
    bg_add = tf.add(bg_mul, sliced_fg, name='bg-add')
    clipped = tf.clip_by_value(bg_add, -1.0, 1.0, name='clipped')
    
    return wav_, fg_volume_, sample_rate_, padding_, offset_, bg_, bg_volume_, clipped