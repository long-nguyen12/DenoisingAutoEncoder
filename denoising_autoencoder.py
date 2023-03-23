from time import time
import keras
from keras import backend as K
import numpy as np
import os
import matplotlib.pylab as plt
from PIL import Image as pil_image
import keras.layers as layers
import keras.models as models
from keras.initializers import orthogonal
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam


class AutoEncoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def conv2d_block(self, x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
        prefix = f'block_{block_id}_'
        x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                          kernel_initializer=kernel_init, name=prefix+'conv')(x)
        x = layers.LeakyReLU(name=prefix+'lrelu')(x)
        x = layers.Dropout(0.2, name=prefix+'drop')((x))
        x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
        return x

    def conv2d_transpose_block(self, x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
        prefix = f'block_{block_id}_'
        x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                                   kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
        x = layers.LeakyReLU(name=prefix+'lrelu')(x)
        x = layers.Dropout(0.2, name=prefix+'drop')((x))
        x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
        return x

    def create_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # 256 x 256
        conv1 = self.conv2d_block(
            inputs, filters=64, kernel=3, strides=1, padding='same', block_id=1)
        conv2 = self.conv2d_block(
            conv1, filters=64, kernel=3, strides=2, padding='same', block_id=2)

        # 128 x 128
        conv3 = self.conv2d_block(
            conv2, filters=128, kernel=5, strides=2, padding='same', block_id=3)

        # 64 x 64
        conv4 = self.conv2d_block(
            conv3, filters=128, kernel=3, strides=1, padding='same', block_id=4)
        conv5 = self.conv2d_block(
            conv4, filters=256,  kernel=5, strides=2, padding='same', block_id=5)

        # 32 x 32
        conv6 = self.conv2d_block(
            conv5, filters=512,  kernel=3, strides=2, padding='same', block_id=6)

        # 16 x 16
        deconv1 = self.conv2d_transpose_block(
            conv6, filters=512,  kernel=3, strides=2, padding='same', block_id=7)

        # 32 x 32
        skip1 = layers.concatenate([deconv1, conv5], name='skip1')
        conv7 = self.conv2d_block(
            skip1, 256, 3, strides=1, padding='same', block_id=8)
        deconv2 = self.conv2d_transpose_block(
            conv7, 128, 3, strides=2, padding='same', block_id=9)

        # 64 x 64
        skip2 = layers.concatenate([deconv2, conv3], name='skip2')
        conv8 = self.conv2d_block(
            skip2, 128, 5, strides=1, padding='same', block_id=10)
        deconv3 = self.conv2d_transpose_block(
            conv8, 64, 3, strides=2, padding='same', block_id=11)

        # 128 x 128
        skip3 = layers.concatenate([deconv3, conv2], name='skip3')
        conv9 = self.conv2d_block(
            skip3, 64, 5, strides=1, padding='same', block_id=12)
        deconv4 = self.conv2d_transpose_block(
            conv9, 64, 3, strides=2, padding='same', block_id=13)

        # 256 x 256
        skip3 = layers.concatenate([deconv4, conv1])
        conv10 = layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid',
                               kernel_initializer=orthogonal(), name='final_conv')(skip3)

        return models.Model(inputs=inputs, outputs=conv10)


class Denoising:
    def __init__(self, dataset_path='data/images', batch_size=20, epochs=150, input_shape=256, noise_factor=1):
        # init folders
        self.SAVE_PATH = 'save_path'
        self.LOGS = 'logs'
        self.SAVE_IMAGES = 'save_imgs'
        dirs = [self.SAVE_PATH, self.LOGS, self.SAVE_IMAGES]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

        # init parameters
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = (input_shape, input_shape)
        self.noise_factor = noise_factor

        self.saved_weight = os.path.join(
            self.SAVE_PATH, 'dataweights.{epoch:02d}-{val_acc:.2f}.hdf5')

        # load model
        self.model = AutoEncoder((input_shape, input_shape, 3)).create_model()

    def random_crop(self, img, random_crop_size):
        width, height = img.size  # PIL format
        dx, dy = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img.crop((x, y, x+dx, y+dy))

    def load_img_extended(self, path, grayscale=False, color_mode='rgb', target_size=None,
                          interpolation='nearest'):
        if grayscale is True:
            print('grayscale is deprecated. Please use '
                  'color_mode = "grayscale"')
            color_mode = 'grayscale'
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if color_mode == 'grayscale':
            if img.mode != 'L':
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError(
                'color_mode must be "grayscale", "rbg", or "rgba"')

        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                img = self.random_crop(img, width_height_tuple)
        return img

    def data_generator(self):
        data_gen_args = dict(
            #     featurewise_center=True,
            #     featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.5, 1.2],
            shear_range=0.01,
            horizontal_flip=True,
            rescale=1/255,
            fill_mode='reflect',
            data_format='channels_last')

        data_flow_args = dict(
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode='input')

        return data_gen_args, data_flow_args

    def noisy_generator(self, batches):
        for batch_x, batch_y in batches:
            sigma = np.random.exponential(0.15)
            noise = self.noise_factor * \
                np.random.normal(scale=sigma, size=batch_x.shape)
            batch_noisy = np.clip(batch_x + noise, 0, 1)
            yield (batch_noisy, batch_y)

    def compile(self, optimizer=Adam(lr=0.002)):
        self.model.compile(optimizer=optimizer, loss='mse',
                           metrics=['accuracy'])

    def fit(self):
        data_gen_args, data_flow_args = self.data_generator()
        # load data
        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(**data_gen_args)

        train_batches = train_datagen.flow_from_directory(
            self.dataset_path + '/train',
            **data_flow_args)

        val_batches = val_datagen.flow_from_directory(
            self.dataset_path + '/train',
            **data_flow_args)

        train_noisy_batches = self.noisy_generator(train_batches)
        val_noisy_batches = self.noisy_generator(val_batches)
        modelchk = keras.callbacks.ModelCheckpoint(self.saved_weight,
                                                   monitor='val_acc',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=2)

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.LOGS,
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=True)

        csv_logger = keras.callbacks.CSVLogger(f'{self.LOGS}/keras_log.csv',
                                               append=True)
        self.model.fit_generator(train_noisy_batches,
                                 steps_per_epoch=train_batches.samples // self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=val_noisy_batches,
                                 validation_steps=train_batches.samples // self.batch_size,
                                 callbacks=[
                                     modelchk, tensorboard, csv_logger],
                                 use_multiprocessing=True)

    def evaluate(self):
        data_gen_args, data_flow_args = self.data_generator()
        model = keras.models.load_model(os.path.join(self.SAVE_PATH,
                                                     'phase2_weights.01-0.86.hdf5'))

        test_datagen = ImageDataGenerator(**data_gen_args)

        test_batches = test_datagen.flow_from_directory(
            self.dataset_path + '/test',
            **data_flow_args)
        test_noisy_batches = self.noisy_generator(test_batches)

        X, y = next(test_noisy_batches)

        score = model.evaluate(X, y, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        decoded_imgs = model.predict(X)

        import matplotlib.pyplot as plt

        n = 5

        plt.figure(figsize=(40, 15))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(X[i])
            ax.axis('off')

            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i])
            ax.axis('off')

        plt.show()


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', default='data/images')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--input_shape', default=256, type=int)
    parser.add_argument('--noise_factor', default=1, type=int)
    args = parser.parse_args()
    print(args)

    denoising = Denoising(dataset_path=args.dataset_path, batch_size=args.batch_size, epochs=args.epochs,
                          input_shape=args.input_shape, noise_factor=args.noise_factor)

    # custom data generator
    keras.utils.load_img = denoising.load_img_extended

    # model
    denoising.model.summary()
    denoising.compile()
    t0 = time()
    denoising.fit()
    print('training time: ', (time() - t0))
    
