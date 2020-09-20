#!/usr/bin/env python

# Salva parametros após algumas iterações de SGD por garantia
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os.path as op


def load_saved_params():
    """ 
    Função auxiliar que carrega parametros salvos 
    anteriormente e resseta contador de iterações
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    params_file = "saved_params_%d.npy" % iter
    np.save(params_file, params)
    with open("saved_state_%d.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f,
        x0,
        step,
        iterations,
        postprocessing=None,
        useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implemente o algoritmo de gradiente descendente nessa função
    

    Argumentos:
    f -- A função a ser otimizada, ela deve receber um único argumento 
        e retornar dois outputs, a loss e o gradiente em relação ao argumento
    x0 -- O ponto inicial de onde começar SGD
    step -- tamanho do passo do SGD
    iterations -- número total de iterações que o SGD deve realizar
    postprocessing -- função de pós-processamento para os parâmetros caso necessário.
                      No caso de wrod2vec nós iremos normalizar 
                      os vetores de palavras para possuir comprimento unitârio.
    PRINT_EVERY -- Especifica de quantas em quantas iterações imprimimos a loss

    Return:
    x -- o valor do parametro após o SGD acabar
    """

    # Diminui learning rate após uma série de iterações
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5**(start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    for iter in range(start_iter + 1, iterations + 1):
        # Será util imprimir seu progresso após algumas iterações

        loss = None
        ### Seu Código Aqui (~2 lines)

        ### Seu código acaba aqui

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x**2), x * 2)

    print("Rodandos testes...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("teste 1:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("teste 2:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("teste 3:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("TODOS OS TESTES FORAM PASSADOS")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()
