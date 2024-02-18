#!/usr/bin/env python

import numpy as np
import random


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


def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset):
    """Função de gradiente & Naive (ingênuo) Softmax loss (custo) para modelos word2vec

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
    """Amostra K indices distintos de outsideWordIdx"""

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec, outsideWordIdx, outsideVectors, dataset, K=10
):
    """Função de custo (loss) de amostragem negativa para modelos word2vec

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


def skipgram(
    currentCenterWord,
    windowSize,
    outsideWords,
    word2Ind,
    centerWordVectors,
    outsideVectors,
    dataset,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    """Modelo de skip-gram no word2vec

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


def word2vec_sgd_wrapper(
    word2vecModel,
    word2Ind,
    wordVectors,
    dataset,
    windowSize,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[: int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2) :, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord,
            windowSize1,
            context,
            word2Ind,
            centerWordVectors,
            outsideVectors,
            dataset,
            word2vecLossAndGradient,
        )
        loss += c / batchsize
        grad[: int(N / 2), :] += gin / batchsize
        grad[int(N / 2) :, :] += gout / batchsize

    return loss, grad


def normalizeRows(x):
    """Função de normalização de linhas

    Implementa  uma  função de normalização para as linhas
    de uma matriz para que elas possuam comprimento unitário

    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N, 1)) + 1e-30
    return x


def gradcheck_naive(f, x, gradientText):
    """Gradient check para a função f.
    Arguments:
    f -- uma função que recebe apenas um argumento e retorna a loss e seu gradiente
    x -- o ponto (numpy array) aonde devemos checar o gradiente
    gradientText -- uma string detalhando um contexto sobre o calculo do gradiente

    Nota:
    A checagem de gradiente é um teste sanitário que apenas checa se o gradiente
    e a loss produzida pela sua implementação estão consistentes entre si.
    Passar essa checagem não garante que seus gradiente estão corretos.
    Por exemplo uma implementação que retorne apenas 0 passaria esse teste.
    Aqui há uma explicação detalhada sobre o que  a checagem de gradiente está

    http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Avalia  a função no ponto original
    h = 1e-4  # não altere isso!

    # Itera sobre todos os índices ix em x para checar o gradiente.
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h
        random.setstate(rndstate)
        fxh, _ = f(x)  # computa f(x + h)
        x[ix] -= 2 * h  # restaura valor anterior
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compara gradientes
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check falhou para %s." % gradientText)
            print(
                "Primeiro erro de gradiente encontrado no índice %s do vetor de gradientes"
                % str(ix)
            )
            print("Seu gradiente: %f \t gradiente numérico: %f" % (grad[ix], numgrad))
            return

        it.iternext()  # Passa pra próxima dimensão

    print(
        "Gradient check passou!. Leia a docstring do método `gradcheck_naive`"
        " em utils.gradcheck.py para entender melhor o que a checagem de gradiente faz."
    )


def grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset):
    print("======Casos de teste para Skip-Gram com naiveSoftmaxLossAndGradient ======")

    # primeiro teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "c",
        3,
        ["a", "b", "e", "d", "b", "c"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
    )

    assert np.allclose(
        output_loss, 11.16610900153398
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.26947339, -1.36873189, 2.45158957],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-0.41045956, 0.18834851, 1.43272264],
        [0.38202831, -0.17530219, -1.33348241],
        [0.07009355, -0.03216399, -0.24466386],
        [0.09472154, -0.04346509, -0.33062865],
        [-0.13638384, 0.06258276, 0.47605228],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o primeiro teste!")

    # segundo teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "b",
        3,
        ["a", "b", "e", "d", "b", "c"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
    )
    assert np.allclose(
        output_loss, 9.87714910003414
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [0.0, 0.0, 0.0],
        [-0.14586705, -1.34158321, -0.29291951],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-0.30342672, 0.19808298, 0.19587419],
        [-0.41359958, 0.27000601, 0.26699522],
        [-0.08192272, 0.05348078, 0.05288442],
        [0.6981188, -0.4557458, -0.45066387],
        [0.10083022, -0.06582396, -0.06508997],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o segundo teste!")

    # terceiro teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "a",
        3,
        ["a", "b", "e", "d", "b", "c"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
    )

    assert np.allclose(
        output_loss, 10.810758628593335
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [-1.1790274, -1.35861865, 1.53590492],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-7.96035953e-01, -1.79609012e-02, 2.07761330e-01],
        [1.40175316e00, 3.16276545e-02, -3.65850437e-01],
        [-1.99691259e-01, -4.50561933e-03, 5.21184016e-02],
        [2.02560028e-02, 4.57034715e-04, -5.28671357e-03],
        [-4.26281954e-01, -9.61816867e-03, 1.11257419e-01],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o terceiro teste!")

    print("Você passou todos os 3 testes!")


def grad_tests_negsamp(
    skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient
):
    print("======Skip-Gram com negSamplingLossAndGradient======")

    # primeiro teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "c",
        1,
        ["a", "b"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
        negSamplingLossAndGradient,
    )

    assert np.allclose(
        output_loss, 16.15119285363322
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-4.54650789, -1.85942252, 0.76397441],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-0.69148188, 0.31730185, 2.41364029],
        [-0.22716495, 0.10423969, 0.79292674],
        [-0.45528438, 0.20891737, 1.58918512],
        [-0.31602611, 0.14501561, 1.10309954],
        [-0.80620296, 0.36994417, 2.81407799],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o primeiro teste!")

    # segundo teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "c",
        2,
        ["a", "b", "c", "a"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
        negSamplingLossAndGradient,
    )
    assert np.allclose(
        output_loss, 28.653567707668795
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-6.42994865, -2.16396482, -1.89240934],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-0.80413277, 0.36899421, 2.80685192],
        [-0.9277269, 0.42570813, 3.23826131],
        [-0.7511534, 0.34468345, 2.62192569],
        [-0.94807832, 0.43504684, 3.30929863],
        [-1.12868414, 0.51792184, 3.93970919],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o segundo teste!")

    # terceiro teste
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = skipgram(
        "a",
        3,
        ["a", "b", "e", "d", "b", "c"],
        dummy_tokens,
        dummy_vectors[:5, :],
        dummy_vectors[5:, :],
        dataset,
        negSamplingLossAndGradient,
    )
    assert np.allclose(
        output_loss, 60.648705494891914
    ), "Sua loss não bate com o esperado."
    expected_gradCenterVecs = [
        [-17.89425315, -7.36940626, -1.23364121],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    expected_gradOutsideVectors = [
        [-6.4780819, -0.14616449, 1.69074639],
        [-0.86337952, -0.01948037, 0.22533766],
        [-9.59525734, -0.21649709, 2.5043133],
        [-6.02261515, -0.13588783, 1.57187189],
        [-9.69010072, -0.21863704, 2.52906694],
    ]

    assert np.allclose(
        output_gradCenterVecs, expected_gradCenterVecs
    ), "Seu gradCenterVecs não bate com o esperado."
    assert np.allclose(
        output_gradOutsideVectors, expected_gradOutsideVectors
    ), "Seu gradOutsideVectors não bate com o esperado."
    print("Passou o terceiro teste!")

    print("Você passou todos os 3 testes!")


def test_word2vec():
    """Test the two word2vec implementations, before running on B2W Sentiment Treebank"""
    dataset = type("dummy", (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], [
            tokens[random.randint(0, 4)] for i in range(2 * C)
        ]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient
        ),
        dummy_vectors,
        "naiveSoftmaxLossAndGradient Gradient",
    )
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient
        ),
        dummy_vectors,
        "negSamplingLossAndGradient Gradient",
    )

    grad_tests_negsamp(
        skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient
    )

def main():
    # main pro autograder do coursera, não mexer
    test_word2vec()

if __name__ == "__main__":
    test_word2vec()
