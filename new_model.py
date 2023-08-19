import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import pickle
########################
path='original'
epochs=10
test_ratio=0.2
########################

myList=os.listdir(path)
print(len(myList))
noOfClasses=len(myList)
images=[]
class_no=[]
print("Total no of classes= ",noOfClasses)
print("Importing Classes...")
for x in range(1,noOfClasses+1):
    myPicList = os.listdir(path +"/"+ str(x))
    for y in myPicList:
        curIMG=cv.imread(path + '/' + str(x) + "/" + y)
        curIMG=cv.resize(curIMG,(32,32))
        images.append(curIMG)
        class_no.append(x)
    print(x,end=" ")

print(" ")
images=np.array(images)
class_no=np.array(class_no)

print("Total images in the list:",images.shape[0])
print("Total classified images:",class_no.shape[0])

#split the data

x_train,x_test,y_train,y_test=train_test_split(images,class_no,test_size=test_ratio)
#x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=test_ratio)
print(x_train.shape)

num_of_samples=[]
for x in range(1,noOfClasses+1):
    y=(len(np.where(y_train==x)[0]))
    num_of_samples.append(y)

print(num_of_samples)

plt.figure(figsize=(10,5))
plt.bar(range(1,noOfClasses+1),num_of_samples)
plt.show()
def preprocessing(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)
    img=img/255
    return img
# img=preprocessing(x_train[45])
# cv.imshow("img",img)
# cv.waitKey(0)

x_train=np.array(list(map(preprocessing,x_train)))
x_test=np.array(list(map(preprocessing,x_test)))
#x_val=np.array(list(map(preprocessing,x_val)))


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
#x_val=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
# img=x_train[49]
# cv.imshow("img",img)
# cv.waitKey(0)
print(x_train[49].shape)

# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=False
# )
#
# datagen.fit(x_train)
print(y_train[0])
y_train=to_categorical(y_train-1, noOfClasses)
#y_val=to_categorical(y_val-1,noOfClasses)
y_test=to_categorical(y_test-1,noOfClasses)
print(y_train[0])
def my_model():
    model=Sequential()
  ###################################
    noOfFilters=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode= 500

  ####################################

    model.add(Conv2D(noOfFilters,sizeOfFilter1,input_shape=(32,32,1),activation='relu')) #only for first convolutional layer to mentain input layer size
    #model.add(Activation("relu")) ##activation func
    #model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling


    model.add(Conv2D(noOfFilters,sizeOfFilter1,activation='relu')) #only for first convolutional layer to mentain input layer size
    #model.add(Activation("relu")) ##activation func
    model.add(MaxPooling2D(pool_size=sizeOfPool))


    model.add(Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')) #only for first convolutional layer to mentain input layer size
    #model.add(Activation("relu")) ##activation func
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2,activation='relu'))  # only for first convolutional layer to mentain input layer size
    #model.add(Activation("relu"))  ##activation func
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))



    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    #model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # #Fully connected layer-2
    #
    # model.add(Dense(32))
    # model.add(Activation("relu"))

    #Last Fully Connected layer

    model.add(Dense(9,activation='softmax'))
    #model.add(Activation("softmax"))

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer= "adam", metrics=["accuracy"])


    return model

model=my_model()
# Compile the model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the augmented data
history=model.fit(x_train,y_train,epochs=epochs,validation_split=0.2)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel("Epochs")
plt.show()

model.save("final_model.hdf5")
# pickle_out=open("trained_model2.p",'wb')
# pickle.dump(model,pickle_out)
#pickle_out.close()