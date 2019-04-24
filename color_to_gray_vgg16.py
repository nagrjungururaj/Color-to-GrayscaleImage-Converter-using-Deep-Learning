import numpy as np
from keras.layers import Input,Dense
from keras.datasets import cifar10
from keras.applications import VGG16
from keras.models import Model
from scipy.misc import imsave
import os

classes = 3
# please paste tto your desired path
path = 'C:\\Users\\Admin\\Desktop\\signzy_challenge'

(x_train, _),(x_test, _) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test/255

R = x_test[:,:,:,0]
G = x_test[:,:,:,1]
B = x_test[:,:,:,2]

def generate_ground_truth_images(no_image):
    #actual scales
    x1 = 0.3 #red
    x2 = 0.59 #green
    x3 = 0.11 #blue

    gray = x1*R[no_image,:,:] + x2*G[no_image,:,:] + x3*B[no_image,:,:]
    imsave(os.path.join(path,'gray_gt.png'), gray)
    return gray

def extract_rgb_scales():

    inp = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    model = VGG16(weights='imagenet',include_top=False)(inp)
    model = Dense(classes,activation='softmax')(model)

    vgg16 = Model(inp,model)
    x = vgg16.predict(x_test)
    x = np.reshape(x,(x_test.shape[0],classes))
    return x

def view_grayscale_images(no_image):
    x = extract_rgb_scales()
    #view the grayscale for 50th test image for example
    red = R[no_image,:,:]
    green = G[no_image,:,:]
    blue = B[no_image,:,:]
    #get the corresponding scales for R,G,B
    x1 = x[no_image,0]
    x2 = x[no_image,1]
    x3 = x[no_image,2]

    #save the original image
    org_img = x_test[no_image,:,:,:]

    imsave(os.path.join(path,'color.png'), org_img)

    #apply the general transform
    grayimage = x1*red + x2*green + x3*blue

    imsave(os.path.join(path,'gray.png'),grayimage)
    return grayimage

#example to run the code, 9 specifies the 10th test image. You can change it for different images.
gt = generate_ground_truth_images(9)
op = view_grayscale_images(9)

#calculate correlation-coeffcient between the generated grayscale and ground truth
C = np.corrcoef(op,gt)[1,0]
print('The correlation co-efficient between the generated grayscale and ground truth image is\t',C)
