from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
from keras.initializers import glorot_uniform
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
import  tkinter.messagebox
from tkinter import ttk
import tkinter as tk
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
import matplotlib.pyplot as pyplot
import sys
import numpy as np
from keras import backend as K
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


root = tk.Tk()
root.geometry("620x400")
root.resizable(0, 0)

img_width, img_height = 227, 227
IMAGE_SIZE = [227, 227]

nb_train_samples = 5000
nb_validation_samples = 600
nb_test_samples = 800
batch_size = 40

y_scrollbar = Scrollbar(root, orient="vertical")
y_scrollbar.pack(side=RIGHT, fill=Y)

x_scrollbar = Scrollbar(root, orient="horizontal")
x_scrollbar.pack(side=BOTTOM, fill=X)

class NewCBox(ttk.Combobox):
    def __init__(self, master, dictionary, *args, **kw):
        ttk.Combobox.__init__(self, master, values=sorted(list(dictionary.keys())), state='readonly', *args, **kw)
        self.dictionary = dictionary
        self.current(0)

    def value(self):
        return self.dictionary[self.get()]

class dataset():
    def __init__(self, train, test, val):
        self.train = train
        self.test = test
        self.val = val

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((320, 320), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=250, y=40)

    def test():
        data_dir = x
        img_pred = image.load_img(data_dir, target_size=(227, 227))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        if newcb.value() == 'A':
            loaded_model = tf.keras.models.load_model("Alexnet.h5", custom_objects={'GlorotUniform': glorot_uniform()})
        elif newcb.value() == 'B':
            loaded_model = tf.keras.models.load_model("V16.h5", custom_objects={'GlorotUniform': glorot_uniform()})
        else:
            loaded_model = tf.keras.models.load_model("V3.h5", custom_objects={'GlorotUniform': glorot_uniform()})
        rslt = loaded_model.predict(img_pred)

        if rslt[0][0] >0.5:
            prediction = ("%.2f" % (rslt[0][0] * 100) + "% Fake")
        else:
            prediction = ("%.2f" % ((1 - rslt[0][0]) * 100) + "% Live")
        return prediction

    def clicked():
        x = test()
        label = Label(root, text=x, font=("Times New Roman", 14),fg= "#00FFBF" )
        label.place(x=350, y=0)

    btnKiemTra = Button(root, text="kiem tra", command=clicked, width=12)
    btnKiemTra.place(x=125, y=170)

"""---------dataset-----------------------"""
data = dataset('Images/training_imgs',"Images/testing_imgs","Images/validation_imgs")
data2 = dataset('Images/training_imgs',"Images/testing_imgs","Images/validation_imgs")

def Take_input():
    INPUT = int(inputtxt.get("1.0", "end-1c"))
    return INPUT

"""--------------mô hình---------------"""
def Alex_net():
    if cbbtrain.value() == "Y":
        train_data_dir = data.train
        validation_data_dir = data.val
        test_data_dir = data.test
    else:
        train_data_dir = data2.train
        test_data_dir = data2.test
        validation_data_dir = data2.val
    epochs = Take_input()
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    # Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))
    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(2))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    train_datagen = ImageDataGenerator()

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size, class_mode='binary')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size, class_mode='binary')
    history = model.fit(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_train_samples // batch_size)

    def summarize_diagnostics(history):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        pyplot.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['accuracy'], color='blue', label='train')
        pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        filename = sys.argv[0].split('/')[-1]
        pyplot.savefig(filename + '_plot1.png')
        pyplot.close()

    model.save("model1.h5")
    print("Saved model to disk")

    score = model.evaluate_generator(test_generator, nb_test_samples/batch_size, workers=12)

    scores = model.predict_generator(test_generator, nb_test_samples/batch_size, workers=12)

    # print("Loss: ", score[0], "Accuracy: ", score[1])

    correct = 0
    for i, n in enumerate(test_generator.filenames):
        if n.startswith("Fake") and scores[i][0] <= 0.5:
            correct += 1
        if n.startswith("Live") and scores[i][0] > 0.5:
            correct += 1

    print("Correct:", correct, " Total: ", len(test_generator.filenames))
    print("Loss: ", score[0], "Accuracy: ", score[1])

    Y_pred = model.predict_generator(test_generator, nb_test_samples // batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Live', 'Fake']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

    summarize_diagnostics(history)

def Vgg():
    if cbbtrain.value() == "Y":
        train_data_dir = data.train
        validation_data_dir = data.val
        test_data_dir = data.test
        folders = glob("C:/Users/Administrator/PycharmProjects/pythonProject/Fake_Fingerprints/Image/Train/*")
    else:
        train_data_dir = data2.train
        test_data_dir = data2.test
        validation_data_dir = data2.val
        folders = glob("C:/Users/Administrator/PycharmProjects/pythonProject/Fake_Fingerprints/Image/Train/*")

    epochs = Take_input()
    V16 = VGG16(input_shape=IMAGE_SIZE + [3], weights=None, include_top=False)

    # don't train existing weights
    for layer in V16.layers:
        layer.trainable = False
    # our layers - you can add more if you want
    x = Flatten()(V16.output)
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    model = Model(inputs=V16.input, outputs=prediction)

    # view the structure of the model

    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator()

    Val_datagen = ImageDataGenerator()

    training_set = train_datagen.flow_from_directory(train_data_dir,
                                                     target_size=IMAGE_SIZE,
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    test_set = Val_datagen.flow_from_directory(validation_data_dir,
                                               target_size=IMAGE_SIZE,
                                               batch_size=batch_size,
                                               class_mode='categorical')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=IMAGE_SIZE,
                                                      batch_size=20, class_mode='categorical')
    # fit the model
    callback = EarlyStopping(monitor='loss', patience=3)
    r = model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=epochs ,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks=[callback]
    )
    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('LossVal_loss')

    # accuracies
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('AccVal_acc')
    model.save('V3.h5')

    score = model.evaluate_generator(test_generator, nb_test_samples / batch_size, workers=12)

    scores = model.predict_generator(test_generator, nb_test_samples / batch_size, workers=12)

    # print("Loss: ", score[0], "Accuracy: ", score[1])

    correct = 0
    for i, n in enumerate(test_generator.filenames):
        if n.startswith("Fake") and scores[i][0] <= 0.5:
            correct += 1
        if n.startswith("Live") and scores[i][0] > 0.5:
            correct += 1

    print("Correct:", correct, " Total: ", len(test_generator.filenames))
    print("Loss: ", score[0], "Accuracy: ", score[1])

    Y_pred = model.predict_generator(test_generator, 800 // 20)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Live', 'Fake']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

def Inception_V3():
    if cbbtrain.value() == "Y":
        train_data_dir = data.train
        validation_data_dir = data.val
        test_data_dir = data.test
        folders = glob("C:/Users/Administrator/PycharmProjects/pythonProject/Fake_Fingerprints/Image/Train/*")
    else:
        train_data_dir = data2.train
        test_data_dir = data2.test
        validation_data_dir = data2.val
        folders = glob("C:/Users/Administrator/PycharmProjects/pythonProject/Fake_Fingerprints/Image/Train/*")

    epochs = Take_input()
    V3 = Inception_V3(input_shape=IMAGE_SIZE + [3], weights=None, include_top=False)

    # don't train existing weights
    for layer in V3.layers:
        layer.trainable = False

        # useful for getting number of classes

    # our layers - you can add more if you want
    x = Flatten()(V3.output)
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    model = Model(inputs=V3.input, outputs=prediction)

    # view the structure of the model

    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator()

    Val_datagen = ImageDataGenerator()

    training_set = train_datagen.flow_from_directory(train_data_dir,
                                                     target_size=IMAGE_SIZE,
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    test_set = Val_datagen.flow_from_directory(validation_data_dir,
                                               target_size=IMAGE_SIZE,
                                               batch_size=batch_size,
                                               class_mode='categorical')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=IMAGE_SIZE,
                                                      batch_size=20, class_mode='categorical')
    # fit the model
    callback = EarlyStopping(monitor='loss', patience=3)
    r = model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=epochs ,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks=[callback]
    )
    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('LossVal_loss')

    # accuracies
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('AccVal_acc')
    model.save('V3.h5')

    score = model.evaluate_generator(test_generator, nb_test_samples / batch_size, workers=12)

    scores = model.predict_generator(test_generator, nb_test_samples / batch_size, workers=12)

    # print("Loss: ", score[0], "Accuracy: ", score[1])

    correct = 0
    for i, n in enumerate(test_generator.filenames):
        if n.startswith("Fake") and scores[i][0] <= 0.5:
            correct += 1
        if n.startswith("Live") and scores[i][0] > 0.5:
            correct += 1

    print("Correct:", correct, " Total: ", len(test_generator.filenames))
    print("Loss: ", score[0], "Accuracy: ", score[1])

    Y_pred = model.predict_generator(test_generator, 800 // 20)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Live', 'Fake']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

"""kiem tra mô hình"""
def click():
    if newcb.value() == 'A':
        Alex_net()
    elif newcb.value() == 'B':
        Vgg()
    else:
        Inception_V3()

"""---------------------------training----------------------"""
lbltrain = Label(root, text="TRAIN:", font = ("Times New Roman", 18))
inputtxt = Text(root, height=1, width=12, bg="light yellow")
Display = Button(root, width=12, text="tranning",command=lambda: click())
lblep = Label(root, text="Nhập epoch:", font=("Times New Roman", 12))
lbltraining = Label(root, text="Train mô hình:", font=("Times New Roman", 12))
lookuptr = {'bộ 1': 'A', 'bộ 2': 'B'}
cbbtrain = NewCBox(root, lookuptr,width= 12)
lblchonbo = Label(root, text="Chọn bộ DL:", font = ("Times New Roman", 12))

lbltrain.place(x=5, y=210)
lblep.place(x=20, y=250)
inputtxt.place(x=125,y=250)
lblchonbo.place(x=20 ,y =290)
cbbtrain.place(x=125,y=290)
lbltraining.place(x=20, y=330)
Display.place(x=125, y=330)
"""----------------option----------------"""
lbloption = Label(root, text="OPTION:", font = ("Times New Roman", 18))
lbloption1 = Label(root, text="Chọn Mô Hình:", font = ("Times New Roman", 12))

lookup = {'Alex.net': 'A', 'VGG-16': 'B', 'Inception_V3' :'C'}
newcb = NewCBox(root, lookup,width= 12)

lbloption.place(x=5, y=10)
lbloption1.place(x=20, y=50)
newcb.place(x=125,y=50)
"""----------------test----------------"""
lbltest= Label(root, text="TEST:", font = ("Times New Roman", 18))
btnopen = Button(root, text='open image', command=open_img,width= 12)
lblopen =Label(root, text="Chọn Ảnh Test:", font=("Times New Roman", 12))
lblkt = Label(root, text="Kiểm Tra Ảnh:", font=("Times New Roman", 12))
lblnope = Label(root, text="Ko Có Ảnh", font=("Times New Roman", 12))

btnopen.place(x=125, y=130)
lblopen.place(x=20, y=130)
lbltest.place(x=5, y=90)
lblkt.place(x=20 ,y=170)
lblnope.place(x=125, y =170)
root.mainloop()

