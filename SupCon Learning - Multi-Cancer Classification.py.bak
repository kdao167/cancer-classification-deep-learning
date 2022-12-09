#!/usr/bin/env python
# coding: utf-8

# In[72]:


import math
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.applications.efficientnet as efn
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import time
import warnings
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import mixed_precision
from tensorflow.keras import optimizers, losses, metrics, Model

# Initializes the seed for the project to make the model deterministic
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# In[73]:


# Lists the GPU's recognized. Should have at least one GPU with CUDA compatibility
# for increased training speed.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
print(physical_devices)
print(tf.test.is_built_with_cuda())


# In[74]:


# Sets the distribution strategy for GPU 
strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE

# Modern GPUs compute faster in 16-bit. Setting this mixed precision policy will increase computation speed.
mixed_precision.set_global_policy('mixed_float16')

# Enables auto-clustering using XLA to increase computation speed.
tf.config.optimizer.set_jit("autoclustering")


# In[75]:


# IMPORTANT: Set global parameters for the model's training and configuration
BATCH_SIZE = 32
PROCESS_HEIGHT = 256
PROCESS_WIDTH = 256
CHANNELS = 3
CLASS_AMT = 26
FOLD_AMT = 5
FOLD_NUM = 1
EPOCHS = 10
HEIGHT = 512
WIDTH = 512
CUT_OUT_PROB = .5
LEARNING_RATE = 3e-5


# Load Data

# In[76]:


# Retrieves the number of data files from the directory
def data_info_grabber(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    total = 0
    for i in n:
        total += i
    return total

# Show a brief example of the dataset
def display_data(data_files):
    display(train.head())
    print(f'Train samples: {len(train)}')


# In[77]:


train = pd.read_csv(f'TFRecordGeneration.csv')

# Dataset paths
CLASSES = ['all_benign',
           'all_early',
           'all_pre',
           'all_pro',
           'brain_glioma',
           'brain_menin',
           'brain_tumor',
           'breast_benign',
           'breast_malignant',
           'cervix_dyk',
           'cervix_koc',
           'cervix_mep',
           'cervix_pab',
           'cervix_sfi',
           'colon_aca',
           'colon_bnt',
           'kidney_normal',
           'kidney_tumor',
           'lung_aca',
           'lung_bnt',
           'lung_scc',
           'lymph_cll',
           'lymph_fl',
           'lymph_mcl',
           'oral_normal',
           'oral_scc']

# Dataset TFRecords
data_files = tf.io.gfile.glob('*.tfrec')
display_data(data_files)

FILENAMES_COMP = data_files
TRAINING_FILENAMES = (FILENAMES_COMP)
TRAIN_IMG_AMT = data_info_grabber(TRAINING_FILENAMES)


# Augmentation

# In[78]:


# Data Augmentation Auxiliary Functions
# 

def randomize_parameters():
    return tf.random.uniform([], 0, 1.0, dtype=tf.float32)

def transform_rotation(image, height, rotation):
    DIM = dimension_fixer(height)
    Fixed_DIM = DIM % 2
    
    rotation*= tf.random.uniform([1],dtype='float16')
    # CONVERT DEGREES TO RADIANS
    rotation = to_radian(rotation)
    
    # ROTATION MATRIX
    s1 = tf.math.sin(rotation)
    c1 = tf.math.cos(rotation)
    one = rotation_matrix(1)
    zero = rotation_matrix(0)
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3])
    dim_t = DIM // 2
    x, y = tf.repeat(tf.range(dim_t,-dim_t,-1), DIM), tf.tile(tf.range(-dim_t,dim_t),[DIM])
    z = tf.ones([DIM*DIM],dtype='int32')
    stacker = [x,y,z]                      
    idx = tf.stack(stacker)    
    transformer1 = K.cast(K.dot(rotation_matrix,tf.cast(idx,dtype='float16')),dtype='int32')
    transformer1 = K.clip(transformer1,-dim_t+Fixed_DIM+1,dim_t)
    idx3 = tf.stack( [dim_t-transformer1[0,], dim_t-1+transformer1[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
    reshaped_val = tf.reshape(d,[DIM,DIM,3])
    return reshaped_val

def data_augment_cutout(image, min_mask_size=(int(HEIGHT * .1), int(HEIGHT * .1)), 
                        max_mask_size=(int(HEIGHT * .125), int(HEIGHT * .125))):
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p1, p2, p3 = .85,.6,.25
    if p_cutout > p1: # 10~15 cut outs
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=tf.random.uniform([], 10, 15, dtype=tf.int32))
    elif p_cutout > p2: # 5~10 cut outs
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=tf.random.uniform([], 5, 10, dtype=tf.int32))
    elif p_cutout > p3: # 2~5 cut outs
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=tf.random.uniform([], 2, 5, dtype=tf.int32))
    else: # 1 cut out
        image = random_cutout(image, HEIGHT, WIDTH, 
                              min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=1)

    return image

def random_cutout(image, height, width, channels=3, min_mask_size=(10, 10), max_mask_size=(80, 80), k=1):
    for i in range(k):
      mask_height = tf.random.uniform(shape=[], minval=min_mask_size[0], maxval=max_mask_size[0], dtype=tf.int32)
      mask_width = tf.random.uniform(shape=[], minval=min_mask_size[1], maxval=max_mask_size[1], dtype=tf.int32)

      pad_h = height - mask_height
      pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
      pad_bottom = pad_h - pad_top

      pad_w = width - mask_width
      pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
      pad_right = pad_w - pad_left

      cutout_area = tf.zeros(shape=[mask_height, mask_width, channels], dtype=tf.uint8)

      cutout_mask = tf.pad([cutout_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
      cutout_mask = tf.squeeze(cutout_mask, axis=0)
      image = tf.multiply(tf.cast(image, tf.float32), tf.cast(cutout_mask, tf.float32))

    return image

def dimension_fixer(height): 
    const = 1 #hyperparameter to adjust 
    return const * height
def to_radian(degree):
    return math.pi * degree / 180
def rotation_matrix(number): 
    return tf.constant([number],dtype='float16')
def transform_shear(image, height, shear):
    DIM = dimension_fixer(height)
    Fixed_DIM = DIM % 2
    shear = tf.random.uniform([1],dtype='float16')
    shear = to_radian(shear)
    one = rotation_matrix(1)
    zero = rotation_matrix(0)
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3])    
    dim_t = DIM // 2
    x, y = tf.repeat( tf.range(dim_t2,-dim_t,-1), DIM ), tf.tile( tf.range(-dim_t,dim_t),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    stacker = [x,y,z]
    idx = tf.stack(stacker)
    transformer1 = K.cast(K.dot(shear_matrix,tf.cast(idx,dtype='float16')), dtype='int32')
    transformer1 = K.clip(transformer1,-dim_t +Fixed_DIM+1,dim_t)    
    # FIND ORIGIN PIXEL VALUES 
    idx3 = tf.stack( [dim_t -transformer1[0,], dim_t -1+transformer1[1,]] )
    d = tf.gather_nd(image, tf.transpose(idx3))
    reshaped_val = tf.reshape(d,[DIM,DIM,3])
        
    return reshaped_val


# In[79]:


# Data Augmentation Functions that utilize the auxiliary functions above
# Functions: Degree Rotation, Pixel Edits, Random Rotation, Shear, Flip, Cut Out

def degree_rotate(image, rotate):
    if rotate > .75:
        image = tf.image.rot90(image, k=3) # rotate 270º
    elif rotate > .5:
        image = tf.image.rot90(image, k=2) # rotate 180º
    elif rotate > .25:
        image = tf.image.rot90(image, k=1) # rotate 90º
        
def pixel_transform(pixel_1, pixel_2, pixel_3, image):
    if p1 < .5:
        image = tf.image.random_saturation(image, lower=.5, upper=1.2)
    if p2 < .5:
        image = tf.image.random_contrast(image, lower=.6, upper=1.1)
    if p3 < .5:
        image = tf.image.random_brightness(image, max_delta=.1)

def rotate_image(image, rotation):
    if rotation > .2:
        if rotation <= .6:
            negative_transform = -1
        image = transform_rotation(image, HEIGHT, rotation= negative_transform * 45.)
        
def shear_image(image, shear):
    if shear > .3:
        if shear <= .65:
            negative_transform = -1
        image = transform_shear(image, HEIGHT, shear=negative_transform * 20.)
        
def flip_image(image, spatial):
    image = tf.image.random_flileft_right(image)
    image = tf.image.random_fliudown(image)
    if spatial > .75:
        image = tf.image.transpose(image)


# In[80]:


# Performs data augmentation using the functions above.
# Functions: Degree Rotation, Pixel Edits, Random Rotation, Shear, Flip, Cut Out

def data_augment(image, label):
    rotation, spatial, rotate, p1, p2, p3, shear, crop, cutout = randomize_parameters(), randomize_parameters(), randomize_parameters(), randomize_parameters(), randomize_parameters(), randomize_parameters(), randomize_parameters()
    negative_transform = 1
    degree_rotate(image, rotate)
    pixel_transform(p1, p2, p3, image)
    rotate_image(image, rotation)
    hear_image(image, shear)
    flip_image(image, spatial)
        
    # Resize to augment
    image = tf.image.resize(image, size=[HEIGHT, WIDTH])

    #Removes certain pixels from image
    if cutout > CUT_OUT_PROB:
        image = data_augment_cutout(image)
        
    return image, label


# In[81]:


# Datasets utility functions
def scale_image(image, label):
    """
        Cast tensor to float and normalizes (range between 0 and 1).
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

def prepare_image(image, label):
    """
        Resize and reshape images to the expected size.
    """
    image = tf.image.resize(image, [PROCESS_HEIGHT, PROCESS_WIDTH])
    image = tf.reshape(image, [PROCESS_HEIGHT, PROCESS_WIDTH, 3])
    return image, label

def read_tfrecord(example, labeled=True):
    """
        1. Parse data based on the 'TFREC_FORMAT' map.
        2. Decode image.
        3. If 'labeled' returns (image, label) if not (image, name).
    """
    if labeled:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'target': tf.io.FixedLenFeature([], tf.int64), 
        }
    else:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'image_name': tf.io.FixedLenFeature([], tf.string), 
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    if labeled:
        label_or_name = tf.cast(example['target'], tf.int32)
        # One-Hot Encoding needed to use "categorical_crossentropy" loss
#         label_or_name = tf.one_hot(tf.cast(label_or_name, tf.int32), N_CLASSES)
    else:
        label_or_name = example['image_name']
    return image, label_or_name

def get_dataset(FILENAMES, labeled=True, ordered=False, repeated=False, 
                cached=False, augment=False):
    """
        Return a Tensorflow dataset ready for training or inference.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        dataset = tf.data.Dataset.list_files(FILENAMES)
        dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
    else:
        dataset = tf.data.TFRecordDataset(FILENAMES, num_parallel_reads=AUTO)
        
    dataset = dataset.with_options(ignore_order)
    
    dataset = dataset.map(lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTO)
    
    if augment:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        
    dataset = dataset.map(scale_image, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_image, num_parallel_calls=AUTO)
    
    if not ordered:
        dataset = dataset.shuffle(2048)
    if repeated:
        dataset = dataset.repeat()
        
    dataset = dataset.batch(BATCH_SIZE)
    
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


# In[82]:


# Displays the dataset distribution of each class
ds_dist = get_dataset(TRAINING_FILENAMES)
labels_comp = [target.numpy() for img, target in iter(ds_dist.unbatch())]

fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax = sns.countplot(y=labels_comp, palette='viridis')
ax.tick_params(labelsize=16)

plt.show()


# In[83]:


# Set training parameters and creates the learning rate function that 
# utilizes cosine learning rate schedule. A warm-up phase is utilizes 
# to ensure that the pre-trained models we use will not be quickly lost. 
# The lower learning rate will ensure that the weights will become more stable.
lr_start = 1e-8
lr_min = 1e-8
lr_max = LEARNING_RATE
num_cycles = 1.
warmup_epochs = 1
hold_max_epochs = 0
total_epochs = EPOCHS
warmup_steps = warmup_epochs * (TRAIN_IMG_AMT//BATCH_SIZE)
hold_max_steps = hold_max_epochs * (TRAIN_IMG_AMT//BATCH_SIZE)
total_steps = total_epochs * (TRAIN_IMG_AMT//BATCH_SIZE)

@tf.function
def cosine_schedule_with_warmup(step, total_steps, warmup_steps=0, hold_max_steps=0, 
                                lr_start=1e-4, lr_max=1e-3, lr_min=None, num_cycles=0.5):
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(np.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, float(lr))

    return lr


# Model

# In[84]:


# Create the encoder model utilizing EfficientNetB3 with imagenet weights
def encoder_fn(input_shape):
    inputs = L.Input(shape=input_shape, name='inputs')
    base_model = efn.EfficientNetB3(input_tensor=inputs, 
                                    include_top=False, 
                                    weights='imagenet',
                                    pooling='avg')
    
    model = Model(inputs=inputs, outputs=base_model.outputs)

    return model

# Create the classifer model utilizing a pre-trained encoder and DenseNet model.
def classifier_fn(input_shape, N_CLASSES, encoder, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable
        
    inputs = L.Input(shape=input_shape, name='inputs')
    
    features = encoder(inputs)
    features = L.Dropout(.5)(features)
    features = L.Dense(1000, activation='relu')(features)
    features = L.Dropout(.5)(features)
    outputs = L.Dense(N_CLASSES, activation='softmax', name='outputs', dtype='float32')(features)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# Supervised Contrastive learning parameters

# In[85]:


# Auxiliary functions for Supervised Contrastive Learning Model

temperature = 0.1

# Supervised Contrastive Loss Function that utilizes labeled information
class SupervisedContrastiveLoss(losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Projection head that projects the encoder's ouput into classifier model.
def add_projection_head(input_shape, encoder):
    inputs = L.Input(shape=input_shape, name='inputs')
    features = encoder(inputs)
    outputs = L.Dense(128, activation='relu', name='projection_head', dtype='float16')(features)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Trains the encoder
def train_encoder():
    print('Pre-training the encoder using "Supervised Contrastive" Loss')
    with strategy.scope():
        encoder = encoder_fn((None, None, CHANNELS))
        encoder_proj = add_projection_head((None, None, CHANNELS), encoder)
        encoder_proj.summary()

        lr = lambda: cosine_schedule_with_warmup(tf.cast(optimizer.iterations, tf.float32), 
                                                 tf.cast(total_steps, tf.float32), 
                                                 tf.cast(warmup_steps, tf.float32), 
                                                 hold_max_steps, lr_start, lr_max, lr_min, num_cycles)
        
        optimizer = optimizers.Adam(learning_rate=lr)
        encoder_proj.compile(optimizer=optimizer, 
                             loss=SupervisedContrastiveLoss(temperature))
        
    history_enc = encoder_proj.fit(x=get_dataset(TRAIN_FILENAMES, repeated=True, augment=True), 
                                   validation_data=get_dataset(VALID_FILENAMES, ordered=True, cached=True), 
                                   steps_per_epoch=step_size, 
                                   batch_size=BATCH_SIZE, 
                                   epochs=EPOCHS,
                                   verbose=2).history
    return encoder

# Trains the classifier using a pre-trained encoder as input
def train_classifier(encoder):
    print('Training the classifier with the frozen encoder')
    warmup_steps = 1
    total_steps = EPOCHS * step_size
    
    with strategy.scope():
        model = classifier_fn((None, None, CHANNELS), N_CLASSES, encoder, trainable=False)
        model.summary()

        lr = lambda: cosine_schedule_with_warmup(tf.cast(optimizer.iterations, tf.float32), 
                                                 tf.cast(total_steps, tf.float32), 
                                                 tf.cast(warmup_steps, tf.float32), 
                                                 hold_max_steps, lr_start, lr_max, lr_min, num_cycles)
        optimizer = optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, 
                      loss=losses.SparseCategoricalCrossentropy(), 
                      metrics=[metrics.SparseCategoricalAccuracy()])
    
    history = model.fit(x=get_dataset(TRAIN_FILENAMES, repeated=True, augment=True), 
                        validation_data=get_dataset(VALID_FILENAMES, ordered=True, cached=True), 
                        steps_per_epoch=step_size, 
                        epochs=EPOCHS,  
                        verbose=2).history
    model_path = f'model_scl_{fold}.h5'
    model.save_weights(model_path)


# Training

# In[86]:


# Train the model:
# First trains encoder, then trains classifier
skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_pred = []; oof_labels = []; oof_embed = []

for fold,(train_split, validation_split) in enumerate(skf.split(np.arange(20))):
    if FOLD_AMT < fold:
        break
    K.clear_session()
    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {train_split} VALID: {validation_split}')

    # Create train and validation sets
    FILENAMES_COMP = tf.io.gfile.glob(['*.tfrec' % x for x in train_split])
    TRAIN_FILENAMES = FILENAMES_COMP
    VALID_FILENAMES =  tf.io.gfile.glob(['*.tfrec' % x for x in validation_split])
    np.random.shuffle(TRAIN_FILENAMES)
       
    step_size = (data_info_grabber(TRAIN_FILENAMES) // BATCH_SIZE)
    warmup_steps = warmup_epochs * step_size
    total_steps = EPOCHS * step_size

    ### Pre-train the encoder
    encoder = train_encoder()

    
    ### Train the classifier with the frozen encoder
    train_classifier(encoder)
    
    ### RESULTS
    print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")

    # OOF predictions
    ds_valid = get_dataset(VALID_FILENAMES, ordered=True)
    oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
    x_oof = ds_valid.map(lambda image, target: image)
    oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))
    oof_embed.append(encoder.predict(x_oof)) # OOF embeddings


# In[ ]:


# Display Out of Fold (OOF) metrics
print(classification_report(y_reg_true, y_reg_pred, target_names=CLASSES))

