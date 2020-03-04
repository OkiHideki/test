# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import time


def flag_search(y_test):
    for index in range(10):
        if y_test[index] == 1:
            return index


def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


if __name__ == '__main__':
    num_classes = 10
    loss = 0
    success_late = 0
    accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    accuracy_ave = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    success_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols, 1)
    # input_shape = (img_rows, img_cols, 1)

    # x_test = x_test.astype(np.float32)  # 非量子化:float32 量子化時:uint8
    x_test = x_test.astype(np.uint8)
    # x_test /= 255.  # 非量子化時
    y_test = keras.utils.to_categorical(y_test, num_classes)

    start = time.time()

    interpreter = tf.lite.Interpreter(model_path="./model/mnist_cnn_tf_quantized.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for num in range(10000):
        print('\nNumber of time: ', num + 1)

        # set input tensor
        interpreter.set_tensor(input_details[0]['index'], x_test[num])

        # run
        interpreter.invoke()

        # get output tensor
        probs = interpreter.get_tensor(output_details[0]['index'])

        # print result
        result = np.argmax(probs[0])
        score = probs[0][result]
        score = score / 255.  # 量子化時
        print("predicted number is {} [{:.2f}]".format(result, score))

        label = flag_search(y_test[num])

        if result == label:
            print('success!')
            success_count[label] += 1

        accuracy[label] += probs[0][label]
        label_count[label] += 1

        # calculate loss and accuracy
        loss += cross_entropy_error(np.array(score), np.array(y_test[num]))
        if num == 9999:
            print('loss = ', loss / 10000)
            print('success_count: ', success_count)
            print("label: ", label_count)

            # for num2 in range(10):
            accuracy = [accuracy[i] / 255 for i in range(10)]  # 量子化時
            accuracy_ave = [accuracy[i] / label_count[i] for i in range(10)]
            print("accuracy: ", accuracy)
            print("accuracy average: ", accuracy_ave)

            """
            for num2 in range(10):
                # accuracy[num2] /= 255  # 量子化時
                accuracy[num2] /= label_count[num2]
                success_late += success_count[num2] / label_count[num2]
                print("accuracy[{}]: {}".format(num2, accuracy[num2]))

                if num2 == 9:
                    success_late /= 10
                    print('success_late = ', success_late)
                    print('sum success count = ', np.sum(success_count))
            """
    process_time = time.time() - start
    print('processing time: ', process_time)