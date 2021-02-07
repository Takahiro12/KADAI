import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import metrics
from tensorflow_addons import optimizers
import sys


class CNN(Model):
    def __init__(self, param):
        super().__init__()
        self.conv1 = Conv2D(filters=8, kernel_size=(5, 5),
                            strides=4, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=2,
                            activation='relu', padding='same')
        self.batch_norm1 = BatchNormalization()
        self.conv3 = Conv2D(filters=32, kernel_size=(
            3, 3), strides=2, activation='relu', padding='same')
        self.batch_norm2 = BatchNormalization()
        self.conv4 = Conv2D(filters=64, kernel_size=(
            3, 3), strides=2, activation='relu', padding='same')
        self.batch_norm3 = BatchNormalization()
        self.avg_pool = GlobalAveragePooling2D()
        self.linear1 = Dense(units=64, activation='relu')
        self.linear2 = Dense(units=param * 3)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = self.avg_pool(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class ImageTransformations:
    def __init__(self, dimension, transformer, func_type, criterion):
        self.dim = dimension

        if transformer == 'channelwise':
            self.model = CNN(self.dim)
            self.transformer = self.channelwise
        else:
            self.transformer = self.fullcolor
            self.model = CNN(self.dim**3)

        if func_type == 'polynomial':
            self.function = self.polynomial
        elif func_type == 'piecewise':
            self.function = self.piecewise
        elif func_type == 'cosine':
            self.function = self.consine
        elif func_type == 'radial':
            self.function = self.radial
        else:
            self.function = self.linear

        if criterion == 'deltaE94':
            self.criterion = self.deltaE94
        else:
            self.criterion = self.deltaE76

        self.optimizer = optimizers.AdamW(
            weight_decay=0.1, learning_rate=0.0001)
        self.epochs = 5000
        self.batch_size = 16

    def BGRToCIELAB(self, img):
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_L, img_a, img_b = cv2.split(img_Lab)
        return img_L, img_a, img_b

    def deltaE76(self, img1, img2):
        L1, a1, b1 = self.BGRToCIELAB(img1)
        L2, a2, b2 = self.BGRToCIELAB(img2)

        loss = 0
        for i in range(img1.shape[0]):
            for j in range(img2.shape[1]):
                loss += np.sqrt((int(L1[i][j])-int(L2[i][j]))**2 +
                                (int(a1[i][j])-int(a2[i][j]))**2+(int(b1[i][j])-int(b2[i][j]))**2)

        return loss

    def deltaE94(self, img1, img2):
        L1, a1, b1 = self.BGRToCIELAB(img1)
        L2, a2, b2 = self.BGRToCIELAB(img2)

        KL = KC = KH = 1
        C1 = np.sqrt(a1**2 - b1**2)
        C2 = np.sqrt(a2**2-b2**2)
        dC = C1 - C2
        dH = np.sqrt((a1-a2)**2 + (b1-b2)**2 + dC**2)
        SL = 1
        K1 = 0.045
        K2 = 0.015
        SC = 1 + K1*C1
        SH = 1 + K2*C1
        dL = np.abs(L1-L2)
        return np.sqrt((dL/(KL*SL))**2 + (dC/(KC*SC))**2 + (dH/(KH*SH))**2)

    def channelwise(self, theta, x):
        theta = tf.reshape(theta, (self.dim, 3))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                term = np.zeros([self.dim, 3])
                for k in range(self.dim):
                    term[k] = theta[k] * self.function(k, x[i][j])
                x[i][j] = x[i][j] + np.sum(term, axis=0)
        return x  # pixel 単位での処理

    def fullcolor(self, theta, x):
        # 各pixel毎に修正
        theta = theta.reshape(self.dim, self.dim, self.dim, 3)
        term = np.zeros(3)

        for c in range(3):
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        term[c] += theta[i][j][k][c] * self.function(
                            i, x[0]) * self.function(j, x[1]) * self.function(k, x[2])
        return x + term

    def polynomial(self, i, x):
        return x**(i)

    def piecewise(self, i, x):
        return np.max(0, 1-np.abs((self.dim-1)*x-i))

    def consine(self, i, x):
        return np.cos(2*np.pi*(i)*x)

    def radial(self, i, x):
        return np.exp(-((x-(i)/(n-1))**2)*self.dim**2)

    def linear(self, i, x):
        return x

    def compute_loss(self, img1, img2):
        return self.criterion(img1, img2)

    def train_step(self, imgs_1_256, imgs_1, imgs_2):
        with tf.GradientTape() as tape:
            theta = self.model(imgs_1_256)
            loss = 0
            for i, _ in enumerate(imgs_1):
                preds = self.transformer(theta[i], imgs_1[i])
                loss += self.compute_loss(preds, imgs_2[i])
            loss = loss / len(imgs_1)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model_trainable_variables))
        return loss

    def fit(self, train_files):
        n_batches = train_files.shape[0] // self.batch_size
        print(n_batches)

        for epoch in range(self.epochs):
            for batch in range(n_batches):
                start = batch * self.batch_size
                end = start + self.batch_size
                batch_files = train_files[start:end]

                imgs_1 = []
                imgs_2 = []
                imgs_1_256 = []
                for file1, file2 in batch_files:
                    img1 = cv2.imread(file1)
                    img2 = cv2.imread(file2)
                    imgs_1.append(img1)
                    imgs_2.append(img2)
                    tmp = cv2.resize(img1.copy(), (256, 256))/255
                    imgs_1_256.append(tmp)

                imgs_1_256 = np.array(imgs_1_256)
                loss = self.train_step(imgs_1_256, imgs_1, imgs_2)
                print(loss)

            print(f'epoch: {epoch}, loss:{loss}')


def main():
    dir_0 = "./FiveK_Lightroom_Export/"
    dir_1 = "./FiveK_artistsC_Export/"

    files_0 = os.listdir("./FiveK_Lightroom_Export/")
    files_1 = [dir_1 + file.split('.')[0] + '.jpg' for file in files_0]
    files_0 = [dir_0 + file for file in files_0]

    train_files, test_files = train_test_split(
        np.array([files_0, files_1]).T, test_size=0.99)

    n = 4
    transformer = 'channelwise'
    func_type = 'polynomial'
    criterion = 'deltaE76'

    obj = ImageTransformations(n, transformer, func_type, criterion)
    obj.fit(train_files)


if __name__ == '__main__':
    main()
