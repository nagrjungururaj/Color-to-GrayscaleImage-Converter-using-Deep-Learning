# Grayscale-to-Color-Image-Converter-using-Deep-Learning
Input/s : Color image/s or RGB images ; 
Ouput/s : Grayscale image/s of single intensity channel

Motivation of Approach : 

1. As this is a type of image reconstruction problem, my first instinct was to use convolutional autoencoders. I tried it but I could not do it due to primary reason that I do not have a GPU to train many parameters on laptop. I know for sure this approach works.

2. Another way of seeing the problem is, the resultant grayscale image is some kind of transformation of the single R,G and B channels. One common approach is to average the channels but these do not produce sharp images. For this, a different method of weighted sums of R,G,B channels are employed

				Gray = 0.3*R + 0.59*G + 0.11*G

Notice that green having less wavelength is given more importance and hence this equation

3. I suggest to learn these scales from a deep neural net and apply the learned scales to the dataset to produce grayscale images.

Solution : 

I think, there are two ways of doing this:

Method 1:

1. Find the wavelengths of each channels in the dataset images which would result in producing 3 labels for each image
2. Design a deep neural net with 3 classes train and test, find the best learned 3 scales 
3. Apply the transformation to produce the gray image.

Disadvantages of this method:

1. Requires good computing power Eg: GPU to train these deep networks and unfortunately my CPU was not able to train even a simple autoencoder.

2. Labelling the images to find the relevance of each channel in the image is complex! 

Method 2: 

1. Use pre-training and a powerful classifier like VGG16 
2. I use the VGG16 pre-trained model extract features from test data and attach a 3 layered simple feed forward neural network to generate the scales for R,G,B channels.
3. I achieve around 65% similarity between the ground truth and the generated grayscale image.
4. Also, I use CIFAR10 dataset for the task. I used this because intially I was going to train the network but later realized its not possible on my laptop.

Disadvantages of this method:
1. As there is no training, the quality of gray image is decent and not the best (does the job) compared to ground truth.

=========================================================================================================================================================================

How to run the code : 

Installables:

1. You can read the guide to install Pycharm here : https://www.jetbrains.com/help/pycharm/install-and-set-up-pycharm.html
2. Install Tensorflow : https://www.tensorflow.org/install
3. Install Keras : https://keras.io/#installation

Note : Install Tensorflow first followed by Keras

1. Unzip the file 
2. Open color_to_gray_vgg16.py in Pycharm.
3. Change paths to save the ground truth image, grayscale and original image in the .py file
4. Run the .py file
