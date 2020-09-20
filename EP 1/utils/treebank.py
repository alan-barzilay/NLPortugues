#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random
import re, string
import pandas as pd


class B2WSentiment:
    def __init__(self, path=None, tablesize=1000000):
        if not path:
            path = "data/B2W-Reviews01.csv"

        self.path = path
        self.df = pd.read_csv(path, nrows=15000, sep=';')[[
            "review_text", "overall_rating"
        ]].drop_duplicates(subset=['review_text'])
        self.tablesize = tablesize

    #################   Detalhes da adaptação   #################
    # Essas funções foram completamente reescritas e a categorify foi removida.
    # getRandomTrainSentence e getSplitSentences foram adaptadas para não usar categorify

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels
        df = self.df
        sent_labels = list(df["overall_rating"].apply(lambda x: x - 1))
        self._sent_labels = sent_labels
        return self._sent_labels

    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        df = self.df

        sentences_raw = list(df["review_text"])

        regex = re.compile("[%s]" % re.escape(string.punctuation))
        sentences_joined = [
            regex.sub(" ", sentence).lower().strip()
            for sentence in sentences_raw
        ]

        sentences = [sentence.split() for sentence in sentences_joined]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def dataset_split(self):
        # 0.7 train
        # 0.2 test
        # 0.1 dev
        if hasattr(self, "_split") and self._split:
            return self._split

        df = self.df

        N = len(df)
        indices = [i for i in range(N)]

        treino = indices[:int(np.floor(N * 0.7))]
        teste = indices[int(np.floor(N * 0.7)):int(np.floor(N * 0.9))]
        dev = indices[int(np.floor(N * 0.9)):]

        split = [treino, teste, dev]

        self._split = split
        return self._split

    #########################################################################################################

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[
            w for w in s if 0 >= rejectProb[tokens[w]]
            or random.random() >= rejectProb[tokens[w]]
        ] for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.sent_labels()[sentId]

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.sent_labels()[i])
                for i in ds_split[split]]

    def sampleTable(self):
        if hasattr(self, "_sampleTable") and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens, ))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq**0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, "_rejectProb") and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens, ))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweight
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
