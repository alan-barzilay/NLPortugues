#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import B2WSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *

# Checa versão de python
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reseta seed aleatória para ter certeza que todos terão os mesmos resultados
random.seed(314)
dataset = B2WSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
# Treinaremos embeddings de 10 dimensões
dimVectors = 10

# Tamanho do contexto
C = 5

# Reseta seed aleatória para ter certeza que todos terão os mesmos resultados
random.seed(31415)
np.random.seed(9265)

startTime = time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors,
     np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(lambda vec: word2vec_sgd_wrapper(
    skipgram, tokens, vec, dataset, C, negSamplingLossAndGradient),
                  wordVectors,
                  0.3,
                  40000,
                  None,
                  True,
                  PRINT_EVERY=100)
# Note que não realizamos normalização aqui. Isso não é um bug,
# normalizar durante treinamento perde an noção de comprimento.

print(
    "Teste sanitário: loss ao final do treinamento deve ser por volta ou menor que 10"
)
print("Treinamento durou %d segundos" % (time.time() - startTime))

# Concatena vetores de input e output
wordVectors = np.concatenate(
    (wordVectors[:nWords, :], wordVectors[nWords:, :]), axis=0)

visualizeWords = [
    "bom",
    "ruim",
    "ótimo",
    "adorei",
    "execelente",
    "exelente",
    "excelente",
    "escelente",
    "excelentes",
    "computador",
    "ventilador",
    "televisão",
    "tv",
    "pessimo",
    "horrivel",
]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i, 0],
             coord[i, 1],
             visualizeWords[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig('vetores_de_palavras.png')
