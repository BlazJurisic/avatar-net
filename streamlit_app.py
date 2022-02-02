from __future__ import absolute_import, division, print_function

import streamlit as st

import os
import time
import scipy.misc
import numpy as np
import tensorflow as tf

from models import models_factory
from models import preprocessing

from PIL import Image

from evaluate_style_transfer import *

import os
import glob
import shutil

slim = tf.contrib.slim

inputs_dir = './img/inputs/'
styles_dir = './img/styles/'
eval_dir = "./img/eval/"

shutil.rmtree(inputs_dir + '.ipynb_checkpoints', ignore_errors=True)
shutil.rmtree(styles_dir + '.ipynb_checkpoints', ignore_errors=True)
shutil.rmtree(eval_dir + '.ipynb_checkpoints', ignore_errors=True)

files = glob.glob(inputs_dir + '*')
if files is not None:
    for f in files:
        os.remove(f)
    
files = glob.glob(styles_dir + '*')
if files is not None:
    for f in files:
        os.remove(f)

files = glob.glob(eval_dir + '*')
if files is not None:
    for f in files:
        shutil.rmtree(eval_dir + '*', ignore_errors=True)

is_input_uploaded = False
is_style_uploaded = False

uploaded_inputs = st.file_uploader("Upload inputs", accept_multiple_files=True)

if uploaded_inputs is not None:
    for uploaded_file in uploaded_inputs:
        bytes_data = uploaded_file.read()
        with open(inputs_dir + uploaded_file.name, 'wb') as fh:
            fh.write(bytes_data)
        is_input_uploaded = True
    
    
uploaded_styles = st.file_uploader("Upload styles", accept_multiple_files=True)
if uploaded_inputs is not None:
    for uploaded_file in uploaded_styles:
        bytes_data = uploaded_file.read()
        with open(styles_dir + uploaded_file.name, 'wb') as fh:
            fh.write(bytes_data)
        is_style_uploaded = True
        
#########################MAIN###################################
################################################################
if is_input_uploaded and is_style_uploaded:
    
    model_config_path = "configs/AvatarNet_config.yml"
    checkpoint_dir = "Avatar-Net/model.ckpt-120000"
    eval_dir = eval_dir
    content_dataset_dir = inputs_dir 
    style_dataset_dir = styles_dir
    model_config_path = "./configs/AvatarNet_config.yml"
    inter_weight = 1.0

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # define the model
        style_model, options = models_factory.get_model(model_config_path)

        # predict the stylized image
        inp_content_image = tf.placeholder(tf.float32, shape=(None, None, 3))
        inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3))

        # preprocess the content and style images
        content_image = preprocessing.mean_image_subtraction(inp_content_image)
        content_image = tf.expand_dims(content_image, axis=0)
        # style resizing and cropping
        style_image = preprocessing.preprocessing_image(
            inp_style_image,
            448,
            448,
            style_model.style_size)
        style_image = tf.expand_dims(style_image, axis=0)

        # style transfer
        stylized_image = style_model.transfer_styles(
            content_image,
            style_image,
            inter_weight=inter_weight)
        stylized_image = tf.squeeze(stylized_image, axis=0)

        # gather the test image filenames and style image filenames
        style_image_filenames = get_image_filenames(style_dataset_dir)
        content_image_filenames = get_image_filenames(content_dataset_dir)

        # starting inference of the images
        init_fn = slim.assign_from_checkpoint_fn(
          checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
        with tf.Session() as sess:
            # initialize the graph
            init_fn(sess)

            nn = 0.0
            total_time = 0.0
            # style transfer for each image based on one style image
            for i in range(len(style_image_filenames)):
                # gather the storage folder for the style transfer
                style_label = style_image_filenames[i].split('/')[-1]
                style_label = style_label.split('.')[0]
                style_dir = os.path.join(eval_dir, style_label)

                if not tf.gfile.Exists(style_dir):
                    tf.gfile.MakeDirs(style_dir)

                # get the style image
                np_style_image = image_reader(style_image_filenames[i])
                print('Starting transferring the style of [%s]' % style_label)

                for j in range(len(content_image_filenames)):
                    # gather the content image
                    np_content_image = image_reader(content_image_filenames[j])

                    start_time = time.time()
                    np_stylized_image = sess.run(stylized_image,
                                                 feed_dict={inp_content_image: np_content_image,
                                                            inp_style_image: np_style_image})
                    incre_time = time.time() - start_time
                    nn += 1.0
                    total_time += incre_time
                    print("---%s seconds ---" % (total_time/nn))

                    output_filename = os.path.join(
                        style_dir, content_image_filenames[j].split('/')[-1])
                    print('Style [%s]: Finish transfer the image [%s]' % (
                        style_label, content_image_filenames[j]))
                    
                    st.write(np_stylized_image.shape)
                    img = np.clip(np_stylized_image, 0, 255).astype(np.uint8)
                    
                    st.image(img)
                    
                    
                    
                    
                    
                    
    
