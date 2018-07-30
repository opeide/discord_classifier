import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K
from datetime import datetime

from model.dataloading import load_data
from model.create_cnn import create_cnn
import os
import glob


if __name__ == '__main__':
    batch_size = 64
    epochs = 20
    n_classes = 2

    #input dimensions
    img_rows, img_cols = 100, 100

    #split data in train and test
    (x_train, y_train), (x_test, y_test) = load_data((3000, 1900), (img_rows,img_cols))
    print(y_test)
    print(x_train.shape)
    if K.image_data_format() == 'channels_first':
        print('channel first')
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        print('channel last')
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test_data samples')


    model = keras.models.load_model('model/pajeet_v1.32')


    # Early stopping callback
    PATIENCE = 5
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

    # TensorBoard callback
    LOG_DIRECTORY_ROOT = '/home/opeide/dev/DL/logs'
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/lastrun/".format(LOG_DIRECTORY_ROOT)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

    # Save after epoch callback
    epochsave = keras.callbacks.ModelCheckpoint('model/epochsave', monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)

    callbacks = [early_stopping, tensorboard, epochsave]

    #clear old run log
    os.remove(glob.glob(log_dir+'*'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/pajeet_v1.33')


