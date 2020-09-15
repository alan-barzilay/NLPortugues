#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Computa a função sigmóide para a entrada.
     Argumentos:
     x - um escalar ou um numpy array
     Retorna:
     s - sigmóide (x)
    """

    ### Seu código aqui (~1 Linha)

    ### Fim do seu código

    return s


def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors,
                                dataset):
    """  Função de gradiente & Naive (ingênuo) Softmax loss (custo) para modelos word2vec

    Implementar a softmax naive loss e gradientes entre o vetor de uma palavra central
    e o vetor de uma palavra externa. Este será o bloco de construção para nossos modelos word2vec.

    Argumentos:
    centerWordVec - numpy ndarray, vetor da palavra central
                    com shape (comprimento do vetor da palavra,)
                    (v_c no enunciado pdf)
    outsideWordIdx - inteiro, o índice da palavra externa
                    (o de u_o no enunciado pdf)
    outsideVectors - matriz de vetores externos com shape (número de palavras no vocabulário, dimensão do embedding)
                    para todas as palavras do vocabulário, cada linha um vetor (U (|V| x n) no folheto em pdf)
    dataset - necessário para amostragem negativa, não utilizado aqui.

    Retorna:
    loss  -  naive softmax loss
    gradCenterVec - o gradiente em relação ao vetor da palavra central
                     com shape (dimensão do embedding,)
                     (dJ / dv_c no enunciado pdf)
    gradOutsideVecs - o gradiente em relação a todos os vetores de palavras externos
                    com shape (num palavras no vocabulário, dimensão do embedding)
                    (dJ / dU)
    """

    ### Seu código aqui (~6-8 Lines)

    ### Use a função softmax fornecida (importada anteriormente neste arquivo)
    ### Esta implementação numericamente estável ajuda a evitar problemas causados
    ### por estouro de inteiro (integer overflow).

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Amostra K indices distintos de outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec,
                               outsideWordIdx,
                               outsideVectors,
                               dataset,
                               K=10):
    """ Função de custo (loss) de amostragem negativa para modelos word2vec

     Implemente o custo de amostragem negativa e gradientes para um vetor de palavra centerWordVec
     e um vetor de palavra outsideWordIdx.
     K é o número de amostras negativas a serem colhidas.

     Observação: a mesma palavra pode ser amostrada negativamente várias vezes. Por
     exemplo, se uma palavra externa for amostrada duas vezes, você deverá
     contar duas vezes o gradiente em relação a esta palavra. Três vezes se
     foi amostrado três vezes e assim por diante.

     Argumentos / especificações de devolução: iguais a naiveSoftmaxLossAndGradient
     """

    # A amostragem negativa de palavras está pronta para você.
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### Seu código aqui  (~10 Lines)

    ### Use sua implementação da função sigmoid aqui.

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord,
             windowSize,
             outsideWords,
             word2Ind,
             centerWordVectors,
             outsideVectors,
             dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Modelo de skip-gram no word2vec

    Implemente o modelo skip-gram nesta função.

    Argumentos:
    currentCenterWord - string da palavra central atual
    windowSize - inteiro, tamanho da janela de contexto
    outsideWords - lista de não mais do que 2 * strings windowSize, as palavras externas
    word2Ind - um objeto dict que mapeia palavras para seus índices 
               na lista de vetores de palavras
    centerWordVectors - matriz dos vetores da palavra central (como linhas) com shape
                        (num palavras no vocabulário, comprimento do vetor da palavra)
                        para todas as palavras do vocabulário ( V no enunciado pdf)
    outsideVectors - matriz dos vetores externos (como linhas) com shape
                        (num palavras no vocabulário, comprimento do vetor da palavra)
                        para todas as palavras do vocabulário (U no enunciado pdf)
    word2vecLossAndGradient - a função de custo e gradiente para
                               um vetor de predição dado os vetores de palavra outsideWordIdx,
                               poderia ser um dos dois funções de perda que você implementou acima.

    Retorna:
    loss - o valor da função de custo para o modelo skipgrama de (J no enunciado pdf)
    gradCenterVec - o gradiente em relação ao vetor da palavra central
                     com shape (comprimento do vetor da palavra,)
                     (dJ / dV no enunciado pdf)
    gradOutsideVecs - o gradiente em relação a todos os vetores de palavras externos
                    com shape (num palavras no vocabulário, comprimento do vetor da palavra)
                    (dJ / dU  no enunciado pdf)
                                        
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### Seu código aqui (~8 Lines)

    ### Seu código acaba aqui

    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# A seguir, funções de teste. NÃO MODIFIQUE #
#############################################


def word2vec_sgd_wrapper(word2vecModel,
                         word2Ind,
                         wordVectors,
                         dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(centerWord, windowSize1, context,
                                     word2Ind, centerWordVectors,
                                     outsideVectors, dataset,
                                     word2vecLossAndGradient)
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print(
        "==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print(
        "==== Gradient check for skip-gram with negSamplingLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset,
                       negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()
