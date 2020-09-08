#!/usr/bin/env python3-

########################################################################
# This file is part of AI Example.
#
# AI Example is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI Example is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
########################################################################


import os
from time import time
import operator
import numpy as np
from tensorflow import keras

"""
Exemple construit sur documentation tensorflow:
    https://www.tensorflow.org/tutorials/keras/classification
"""

epochs = 10


CHARS_DICT = {  "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
                "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13,
                "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
                "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25}


def main(data):
    # Chargement des images
    [train_images, train_labels, test_images, test_labels] = data
    class_names = get_class_names()

    t = time()
    # Construire le réseau de neuronnes nécessite de configurer les couches du
    # modèle et ensuite de le compiler.
    model = build_the_model()
    model = compile_the_model(model)

    # Apprentissage
    model = training_the_model(model, train_images, train_labels, epochs)
    temps = round(time() - t, 1)

    # Test de l'efficacité
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nTesting ......    ")
    print(f"Efficacité sur les tests: {round(test_acc*100, 2)} %")

    # Quelques tests
    print(f"\nQuelques test ...")
    testing_the_model(model, test_images, test_labels)

    print(f"\nApprentissage en {temps} secondes")


def get_data(data_file, train, test):
    """La partie la plus ennuyeuse de l'apprentissage automatique !
    T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
    x_train = data pour training = 16000x16
    y_train = labels = 16000x1
    x_test = data pour testing = 4000x16
    y_test = labels = 4000x1
    """

    print("\nGet datas ...")
    with open(data_file) as f:
        text = f.read()
    f.close()

    # Les datas dans un dict
    #                 1 2 3 4 5 6 7  8 9 10 11 12 13 14 15 16
    # data = {0: [T, [2,8,3,5,1,8,13,0,6, 6,10, 8, 0, 8, 0, 8]]}

    data = {}
    n = 0
    for line in text.splitlines():
        d = line.split(',')
        data[n] = [d[0], []]
        for i in range(1, 17):
            data[n][1].append(int(d[i]))
        n += 1

    # Création des arrays np.zeros((train, 16), dtype=np.uint8)
    xtr, ytr, xte, yte = [],[], [],[]

    # Remplissage des arrays
    i = 0
    for k, v in data.items():

        # Conversion de la lettre en nombre entier = numéro de l'objet
        label = CHARS_DICT[v[0]]

        # Insertion par lignes
        if i < train:
            # les 16 entiers caractérisants la lettre
            xtr.append(v[1])
            # Le numéro de la lettre
            ytr.append(CHARS_DICT[v[0]])
        else:
            # les 16 entiers caractérisants la lettre
            xte.append(v[1])
            # Le numéro de la lettre
            yte.append(CHARS_DICT[v[0]])
        i += 1

    x_train = np.array(xtr)
    y_train = np.array(ytr)
    x_test  = np.array(xte)
    y_test  = np.array(yte)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # (16000, 16) (16000,) (4000, 16) (4000,)

    # Keras veut un array de shape 3 avec GPU! car c'est censé être des images
    x_train = np.expand_dims(x_train, axis=2)
    x_test  = np.expand_dims(x_test, axis=2)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # (16000, 16, 1) (16000,) (4000, 16, 1) (4000,)

    return [x_train, y_train, x_test, y_test]


def get_class_names():
    """Liste des 26 noms d'objets"""

    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return list(L)


def build_the_model():
    """Set up the layers:
        The basic building block of a neural network is the *layer*. Layers
        extract representations from the data fed into them. Hopefully, these
        representations are meaningful for the problem at hand.

        Most of deep learning consists of chaining together simple layers. Most
        layers, such as `tf.keras.layers.Dense`, have parameters that are
        learned during training.

        The first layer in this network, `tf.keras.layers.Flatten`, transforms
        the format of the images
        from a two-dimensional array (of 40 by 40 pixels)
        to a one-dimensional array (of 40 * 40 = 1600 pixels).
        Think of this layer as unstacking rows of pixels in the image and
        lining them up. This layer has no parameters to learn; it only
        reformats the data.

        After the pixels are flattened, the network consists of a sequence
        of two `tf.keras.layers.Dense` layers. These are densely connected,
        or fully connected, neural layers. The first `Dense` layer has 128
        nodes (or neurons).

        The second (and last) layer is a 27-node *softmax* layer that returns
        an array of 27 probability scores that sum to 1. Each node contains
        a score that indicates the probability that the current image belongs
        to one of the 27 classes.
    """

    print("\nBuild the model ...")
    model = keras.Sequential([  keras.layers.Flatten(input_shape=(16, 1)),
                                keras.layers.Dense(128, activation='relu'),
                                keras.layers.Dense(26, activation='softmax') ])
    return model


def compile_the_model(model):
    """Compile the model:
        Before the model is ready for training, it needs a few more settings.
        These are added during the model's *compile* step:

            * *Optimizer*
                This is how the model is updated based on the data it sees and its
                loss function.

            * *Loss function*
                This measures how accurate the model is during training.
                You want to minimize this function to "steer" the model in the
                right direction.

            * *Metrics*
                Used to monitor the training and testing steps. The following
                example uses *accuracy*, the fraction of the images that are
                correctly classified.
    """

    print("\nCompile the model ...")
    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'] )
    return model


def training_the_model(model, train_images, train_labels, epochs):
    """Training the neural network model requires the following steps:

        1. Feed the training data to the model. In this example, the training
        data is in the `train_images` and `train_labels` arrays.
        2. The model learns to associate images and labels.
        3. You ask the model to make predictions about a test set—in this
        example, the `test_images` array. Verify that the predictions match the
        labels from the `test_labels` array.

    To start training, call the `model.fit` method—so called because it "fits"
    the model to the training data:
    """

    print("\nTraining the model ...")
    model.fit(train_images, train_labels, epochs=epochs)
    return model


def testing_the_model(model, test_images, test_labels):
    """It turns out that the accuracy on the test dataset is a little less than the
    accuracy on the training dataset. This gap between training accuracy and test
    accuracy represents *overfitting*. Overfitting is when a machine learning model
    performs worse on new, previously unseen inputs than on the training data.

    Make predictions
        With the model trained, you can use it to make predictions about some images.
    """
    predictions = model.predict(test_images)

    """
    Here, the model has predicted the label for each image in the testing set.
    Let's take a look at the first prediction:
    """

    L = get_class_names()
    print(f"Test sur la 13 ème imagede test: label = {test_labels[13]} soit {L[13]}")
    print(f"    {test_images[13]}")
    print(f"L'indice 13 correspond à la 16014 ème ligne soit L, soit 4,9,4,6,2,0,2,4,6,1,0,7,0,8,0,8 !")

    """
    A prediction is an array of 27 numbers. They represent the model's "confidence"
    that the image corresponds to each of the 27 different objects.
    You can see which label has the highest confidence value:
    """

    print(f"\nPrédiction ...")
    pred = np.argmax(predictions[13])

    print(f"    Prédiction de la 13 ème image: {pred} soit {L[pred]}")

    """
    Finally, use the trained model to make a prediction about a single image.

    `model.predict` returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch

    # `tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list

    """

    print("\nTest sur 10 images")
    for i in range(100):
        img = test_images[2*i]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # Now predict the correct label for this image
        # Retourne une liste de 27 prédictions
        predictions_single = model.predict(img)
        print("Image: {} Prédiction {}".format( test_labels[2*i],
                                                np.argmax(predictions_single[0])))


if __name__ == "__main__":

    data_file = './letter-recognition.data'
    train = 16000
    test = 4000
    data = get_data(data_file, train, test)
    main(data)
