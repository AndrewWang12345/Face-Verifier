import cv
import os
import random
import numpy as np
from matplotlib import pyplot
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D, Input,Flatten
import tensorflow as tf
# gpus=tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
POS_PATH=os.path.join('data','positive')
NEG_PATH=os.path.join('data','negative')
ANC_PATH=os.path.join('data','anchor')
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw',directory)):
#         EX_PATH = os.path.join('lfw',directory,file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH,NEW_PATH)
anchor=tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive=tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative=tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)
def preprocess(file_path):
    byte_img=tf.io.read_file(file_path)
    img=tf.io.decode_jpeg(byte_img)
    img=tf.image.resize(img,(100,100))
    img=img/255.0
    return img
positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data=positives.conatenate(negatives)
samples=data.as_numpy_iterator()
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

# data pipeline
data=data.map(preprocess_twin)
data=data.cache()
data=data.shuffle(buffer_size=1024)

#training partition
train_data=data.take(round(len(data)*0.7))
train_data=train_data.batch(16)
train_data=train_data.prefetch(8)

#test partition
test_data=data.skip(round(len(data)*0.7))
test_data=test_data.take(round(len(data)*0.3))
test_data=test_data.batch(16)
test_data=test_data.prefetch(8)

def make_embedding():
    inp=Input(shape=(100,100,3),name='input_image')
    c1=Conv2D(64,(10,10), activation='relu')(inp)
    m1=MaxPooling2D(64,(2,2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(inp)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    c3=Conv2D(128,(4,4),activation='relu')(m2)
    m3=MaxPooling2D(64,(2,2), padding='same')(c3)

    c4=Conv2D(256,(4,4), activation='relu')(m3)
    f1=Flatten()(c4)
    d1=Dense(4096,activation='sigmoid')(f1)

    return Model(inputs=[inp],outputs=[d1],name='embedding')
embedding=make_embedding()

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)

def make_siamese_model():
    input_image=Input(name='input_img',shape=(100,100,3))

    validation_image=Input(name='validation_img',shape=(100,100,3))

    siamese_layer=L1Dist()
    siamese_layer._name='distance'
    distances=siamese_layer(embedding(input_image),embedding(validation_image))

    classifier=Dense(1,activation='sigmoid')(distances)
    return Model(input=[input_image,validation_image],outputs=classifier, name='Siamesenetwork')
siamese_model=make_siamese_model()
binary_cross_loss=tf.losses.BinaryCrossentropy()
opt=tf.keras.optimizers.Adam(1e-4)

checkpoint_dir='./training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
checkpoint=tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape as tape:
        x=batch[:2]
        y=batch[2]
        yhat=siamese_model(x,training=True)
        loss=binary_cross_loss(y,yhat)
    grad=tape.gradient(loss,siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad),siamese_model.trainable_variables)
    return loss

def train(data, EPOCHS):
    for epoch in range(1,EPOCHS+1):
        print("\n Epoch {}/{}".format(epoch,EPOCHS))
        progbar=tf.keras.utils.Progbar(len(train_data))
        for idx, batch in enumerate(train_data):
            train_step(batch)
            progbar.update(idx+1)
        if epoch%10==0:
            checkpoint.save(file_prefix=checkpoint_prefix)