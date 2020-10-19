from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
Imagesize=128
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(Imagesize,Imagesize,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(11, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 

data_train = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True) 
data_valid = ImageDataGenerator(rescale=1.255)



train_gen = data_train.flow_from_directory('data/train',target_size=(Imagesize,Imagesize),batch_size=108, class_mode='categorical')
valid_gen = data_valid.flow_from_directory('data/valid', target_size=(Imagesize,Imagesize), batch_size=32, class_mode='categorical')

model.fit(train_gen, epochs=10, steps_per_epoch=47, validation_data=valid_gen, validation_steps=7, workers=4,shuffle=True)


model.save('gestures_trained_cnn_model.h5')
