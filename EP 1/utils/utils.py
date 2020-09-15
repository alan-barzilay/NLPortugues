#!/usr/bin/env python

import numpy as np


def normalizeRows(x):
    """ Função de normalização de linhas

    Implementa  uma  função de normalização para as linhas
    de uma matriz para que elas possuam comprimento unitário

    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N, 1)) + 1e-30
    return x


def softmax(x):
    """Computa a função softmax para cada linha do input x.
    É crucial que essa função seja otimizada pois ela será utilizada frequentemente neste programa.

    Argumentos:
    x -- Um vetor de dimensão D ou uma matriz numpy de dimensão NxD
    Return:
    x -- Você pode modificar x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x