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

from time import time
import operator
import numpy as np
import cupy as cp


CHARS_DICT = {  "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
                "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13,
                "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
                "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25}


def sigmoid(x):
    """La fonction sigmoïde est une courbe en S:
    https://fr.wikipedia.org/wiki/Sigmo%C3%AFde_(math%C3%A9matiques)
    """

    return 1 / (1 + cp.exp(-x))

def sigmoid_prime(z):
    """La dérivée de la fonction sigmoid,
    soit sigmoid' comme f' !
    """

    return z * (1 - z)

def relu(x):
    """Rectifie les négatifs à 0:
    -1 > 0
     1 > 1
     Rectified Linear Unit:

    In the context of artificial neural networks, the rectifier is an
    activation function defined as the positive part of its argument.
    https://bit.ly/2HyT4ZO sur Wikipedia en.
     """

    return cp.maximum(0, x)

def relu_prime(z):
    """Fonction: 1 pour tous les réels positifs ou nuls et 0 pour les réels négatifs.

    La fonction de Heaviside (également fonction échelon unité, fonction
    marche d'escalier) est la fonction indicatrice de R.
    Une fonction fonction indicatrice, est une fonction définie sur un
    ensemble E qui explicite l’appartenance ou non à un sous-ensemble F de E
    de tout élément de E.
    """

    return cp.asarray(z > 0, dtype=cp.float32)

def get_data(data_file, train, test):
    """La partie la plus ennuyeuse de l'apprentissage automatique !
    T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
    x_train = data pour training = 16000x16
    y_train = labels = 16000x1
    x_test = data pour testing = 4000x16
    y_test = labels = 4000x1
    """

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

    # Création des arrays
    x_train = np.zeros((train, 16), dtype=np.uint8)
    x_test = np.zeros((test, 16), dtype=np.uint8)
    y_train = np.zeros((train), dtype=np.uint8)
    y_test = np.zeros((test), dtype=np.uint8)

    # Remplissage des arrays
    i = 0
    for k, v in data.items():

        # Conversion de la lettre en nombre entier = numéro de l'objet
        label = CHARS_DICT[v[0]]

        # Les valeurs de la lettre
        values = v[1]

        # Insertion par lignes
        if i < train:
            x_train[i] = values
            y_train[i] =  label
        else:
            x_test[i - train] =  v[1]
            y_test[i - train] =  label
        i += 1

    return [x_train, y_train, x_test, y_test]


class AIExampleCupy:
    """Réseau de neuronnes Perceptron multicouches avec cupy."""

    def __init__(self, data, learningrate):
        """ data_file = fichier des datas
            learningrate = coeff important
        """

        self.learningrate = learningrate

        # Les datas pour cupy, elles sont chargées en RAM GPU
        self.x_train = cp.array(data[0])
        self.y_train = cp.array(data[1])
        self.x_test = cp.array(data[2])
        self.y_test = cp.array(data[3])
        print(type(self.x_train))
        a = "Training: Shot {} Lettre {}; Testing: Shot {} Lettre {}"
        print(a.format( len(self.x_train), len(self.y_train),
                        len(self.x_test),  len(self.y_test)))

        # Réseau de neurones: colonne 16 en entrée, 2 nodes de 100, sortie de 26 caractères
        self.layers = [16, 100, 100, 26]
        # Fonction d'activation: imite l'activation d'un neuronne
        self.activations = [relu, relu, sigmoid]

    def training(self):
        """Apprentissage avec 16 000 lignes"""

        # Matrice diagonale de 1
        diagonale = cp.eye(26, 26)

        # globals() Return a dictionary representing the current global symbol table.
        self.activations_prime = [globals()[fonction.__name__ + '_prime'] \
                                            for fonction in self.activations]

        node_dict = {}

        # Liste des poids
        # Initialisation des poids des nodes, pour ne pas à être à 0
        # Construit 3 matrices (100x1600, 100x100, 27x100)
        # /cp.sqrt() résultat expérimental de l'initialisation de Xavier Glorot et He
        weight_list = [cp.random.randn(self.layers[k+1], self.layers[k]) / \
                       cp.sqrt(self.layers[k]) for k in range(len(self.layers)-1)]

        # vecteur_ligne = image en ligne à la 1ère itération
        # nombre_lettre = nombre correspondant à la lettre de l'image
        # i pour itération, vecteur_colonne = x_train de i, nombre_lettre = y_train de i
        for i, (vecteur_ligne, nombre_lettre) in enumerate(zip(self.x_train, self.y_train)):

            # la ligne devient colonne
            vecteur_colonne = cp.array(vecteur_ligne, ndmin=2).T

            # IndexError: arrays used as indices must be of integer or boolean type.
            # (actual: <class 'numpy.object_'>) in diagonale[:,[nombre_lettre]]
            nombre_lettre = int(nombre_lettre)

            # Forward propagation
            node_dict[0] = vecteur_colonne
            for k in range(len(self.layers)-1):
                # weight_list[k] (100x1600, 100x100 27x100) vecteur_colonne (1600,)
                # z de format 100 x 1
                z = cp.dot(weight_list[k], vecteur_colonne)

                # self.activations = non linéaire sinon sortie fonction linéaire de l'entrée
                # imite le seuil d'activation électrique du neuronne
                vecteur_colonne = self.activations[k](z)

                node_dict[k+1] = vecteur_colonne

            # Retro propagation, delta_a = écart entre la sortie réelle et attendue
            delta_a = vecteur_colonne - diagonale[:,[nombre_lettre]]

            # Parcours des nodes en sens inverse pour corriger proportionnellement
            # les poids en fonction de l'erreur par rapport à la valeur souhaitée
            # Descente du Gradient stochastique
            for k in range(len(self.layers)-2, -1, -1):
                delta_z = delta_a * self.activations_prime[k](node_dict[k+1])
                delta_w = cp.dot(delta_z, node_dict[k].T)
                delta_a = cp.dot(weight_list[k].T, delta_z)
                # Pour converger vers le minimum d'erreur
                weight_list[k] -= self.learningrate * delta_w

        return weight_list

    def testing(self, weight_list):
        """Teste avec les images de testing, retourne le ratio de bon résultats"""

        # Nombre de bonnes reconnaissance
        success = 0

        for vecteur_ligne, nombre_lettre in zip(self.x_test, self.y_test):

            vecteur_ligne = cp.array(vecteur_ligne)
            nombre_lettre = cp.array(nombre_lettre)

            for k in range(len(self.layers)-1):
                vecteur_ligne = self.activations[k](cp.dot(weight_list[k], vecteur_ligne))

            reconnu = cp.argmax(vecteur_ligne)
            if reconnu == nombre_lettre:
                success += 1

        if len(self.x_test) != 0:
            resp = 100.0 * success / len(self.x_test)
        else:
            resp = 0
        return resp


if __name__ == "__main__":

    data_file = './letter-recognition.data'

    train = 16000
    test = 4000
    data = get_data(data_file, train, test)
    print(f"Get data done. {data[0].shape, data[1].shape, data[2].shape, data[3].shape}")
    for i in range(10):
        print(f"Train Value {i} = {data[0][i]}")
        print(f"Train Label {i} = {data[1][i]}")
        print(f"Test Value {i} = {data[2][i]}")
        print(f"Test Label {i} = {data[3][i]}")


    # Parameter Optimization pour le learningrate
    # 0.0222  # meilleur résultat
    t = time()
    result = []
    for k in range(100):
        learningrate = 0.0200 + (k * 0.00005)

        aie = AIExampleCupy(data, learningrate)

        weight_list = aie.training()
        resp = aie.testing(weight_list)
        result.append([learningrate, resp])

        print(f"Learningrate: {learningrate} Résultat {round(resp, 2)} %")
    print("Temps de calcul par cycle:", round((time()-t)/100, 2), "s")

    best = sorted(result, key=operator.itemgetter(1), reverse=True)
    print(f"Meilleur résultat: learningrate={best[0][0]} efficacité={best[0][1]}")
