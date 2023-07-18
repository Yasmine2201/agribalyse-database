# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy
# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 


def crossval(X, Y, n_iterations, iteration):
    """ ndarray * ndarray * int * int -> ndarra,ndarray,ndarray,ndarray
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
        hypothèse : X et Y sont mélangés au départ
    """
    #génère les indices des échantillons test et train
    indices_test=[i for i in range(iteration*len(X)//n_iterations,(iteration+1)*len(X)//n_iterations)]
    indices_train=np.setdiff1d([i for i in range(len(X))],indices_test) 
    
    Xtest=X[indices_test]
    Ytest=Y[indices_test]
    
    Xtrain= X[indices_train]
    Ytrain=Y[indices_train]
    
    return Xtrain,Ytrain,Xtest, Ytest

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    """ ndarray * ndarray * int * int -> ndarray, ndarray, ndarray, ndarray
        La fonction effectue une validation croisée,
        elle sépare les données en n_iterations en conservant la proportion des classes dans chaque séparation
    """
    labels = np.unique(Y)
    liste = []

    # Pour chaque classe
    for i in range(len(labels)):
        #sélection des échantillons de la classe i et application du crossval sur les échantillons de la classe i
        liste.append(crossval(X[Y==labels[i]], Y[Y==labels[i]], n_iterations, iteration))

    # Concaténation des résultats de crossval pour chaque classe
    Xtrain = np.concatenate([liste[i][0] for i in range(len(liste))])
    Ytrain = np.concatenate([liste[i][1] for i in range(len(liste))])
    Xtest = np.concatenate([liste[i][2] for i in range(len(liste))])
    Ytest = np.concatenate([liste[i][3] for i in range(len(liste))])

    return Xtrain, Ytrain, Xtest, Ytest



def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (np.mean(L),np.std(L) )


def validation_croisee(C, DS, nb_iter):
    """ Classifier * tuple[array, array] * int -> tuple[list[float], float, float]
        Effectue une validation croisée pour évaluer les performances d'un classifieur.
        C : Classifieur à évaluer
        DS : Tuple contenant les données d'entrée X et les labels Y
        nb_iter : Nombre d'itérations de la validation croisée
        Renvoie un tuple contenant la liste des performances pour chaque itération, la moyenne des performances et l'écart type des performances.
    """
    X, Y = DS   
    perf = []
    
    for i in range(nb_iter):
        newC = copy.deepcopy(C)
        desc_train,label_train,desc_test,label_test=crossval_strat(X,Y,nb_iter,i)
        newC.train(desc_train,label_train)
        acc_i=newC.accuracy(desc_test,label_test)
        perf.append(acc_i)
        print(i,": taille app.= ",label_train.shape[0],"taille test= ",label_test.shape[0],"Accuracy:",acc_i)
    
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)
