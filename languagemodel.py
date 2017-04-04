# Alex King
# COMP 150NLP
# Final Project

# Definition of bigram language model, tweaked from problem set 2 as necessary

import sys
from collections import defaultdict
from math import log, exp
import pickle
import string
import time

# Global constants

unknown_token = "<UNK>"  # unknown word token
start_token = "<S>"  # sentence boundary token
end_token = "</S>"  # sentence boundary token

LAPLACE_CONSTANT = 10**(-40)

class BigramLM:
    def __init__(self, vocabulary = set()):
        self.vocabulary = vocabulary
        self.unigram_counts = {} 
        self.bigram_counts = {} 
        self.laplace = False
        self.interpolation = False
        self.unigrams_observed = 0
        self.most_common_next_word = {}

    def GetBigramProbability(self, precedent, token):
        """Returns probability of observing token given precedent"""
        if precedent in self.unigram_counts:
            precedent_count = self.unigram_counts[precedent]
        else:
            precedent_count = 0.0

        # If Laplace smoothing is turned on, non-observed bigrams are observed
        # once
        if self.laplace:
            if (precedent in self.bigram_counts and 
                token in self.bigram_counts[precedent]):
                occurrences = (self.bigram_counts[precedent][token] + 
                               LAPLACE_CONSTANT)
            else:
                occurrences = LAPLACE_CONSTANT / 10**(len(token))
            return (occurrences) / (precedent_count + len(self.unigram_counts))

        # If interpolation is turned on, probability is calculated according to
        # lambdas
        elif self.interpolation:
            (lambda1, lambda2) = self.interpolation
            unigram_probability = lambda1 * self.GetUnigramProbability(token)
            
            if (precedent in self.bigram_counts and 
                token in self.bigram_counts[precedent]):
                occurrences = self.bigram_counts[precedent][token]
            else:
                occurrences = 0.0

            bigram_probability = lambda2 * (occurrences / precedent_count)

            return unigram_probability + bigram_probability

        else:
            if (precedent in self.bigram_counts and 
                token in self.bigram_counts[precedent]):
                occurrences = self.bigram_counts[precedent][token]
            else:
                occurrences = 0.0

            return occurrences / precedent_count

    def GetUnigramProbability(self, token):
        """Returns probability of observing token"""

        # If Laplace smoothing is turned on, non-observed unigrams are observed
        # once
        if self.laplace:
            if token in self.unigram_counts:
                occurrences = self.unigram_counts[token] + LAPLACE_CONSTANT
            else:
                # If not seen, scale it down by its length
                occurrences = LAPLACE_CONSTANT  / 10**(len(token))
            return (occurrences / 
                       (self.unigrams_observed + len(self.unigram_counts)))

        else:
            if token in self.unigram_counts:
                occurrences = self.unigram_counts[token]
            else:
                return 0.0
            return occurrences / self.unigrams_observed

    def EstimateBigrams(self, training_data):
        """Creates bigram probability estimates from given training data"""

        unigrams_observed = 0

        # For each unigram and bigram, add them to their respective dictionaries
        for sent in training_data:
            for i in range(len(sent)):

                # Add unigram occurrence
                if sent[i] in self.unigram_counts:
                    self.unigram_counts[sent[i]] += 1.0
                else:
                    self.unigram_counts[sent[i]] = 1.0
                unigrams_observed += 1.0

                # Add bigram occurrence, excepting reaching one past the end
                # of the sentence
                try:
                    if (sent[i] in self.bigram_counts):

                        if sent[i + 1] in self.bigram_counts[sent[i]]:
                            self.bigram_counts[sent[i]][sent[i + 1]] += 1.0
                        else:
                            self.bigram_counts[sent[i]][sent[i + 1]] = 1.0

                    else:
                        self.bigram_counts[sent[i]] = {}
                        self.bigram_counts[sent[i]][sent[i + 1]] = 1.0


                    if sent[i + 1] != unknown_token:
                        # As we go, keep track of most common following words
                        times_seen = self.bigram_counts[sent[i]][sent[i + 1]]
                        if sent[i] in self.most_common_next_word:
                            (word, seen) = self.most_common_next_word[sent[i]]
                            if times_seen > seen:
                                self.most_common_next_word[sent[i]] = \
                                    (sent[i + 1], times_seen)
                        else:
                            self.most_common_next_word[sent[i]] = \
                                (sent[i + 1], times_seen)



                except IndexError:
                    continue

        self.unigrams_observed = unigrams_observed
        return

    def LaplaceSmooth(self, flag):
        """
        Switches on Boolean flag that causes all probabilities to be calculated
        according to Laplace smoothing; turns off interpolation
        """
        self.laplace = flag
        self.interpolation = False
        return

    def Interpolation(self, flag, lambda1 = 0.5, lambda2 = 0.5):
        """
        Updates probabilities according to linear interpolation given lambdas
        """

        if not flag:
            self.interpolation = False
            return

        assert lambda1 + lambda2 == 1

        self.interpolation = (lambda1, lambda2)
        self.laplace = False
        return

    def DeletedInterpolation(self, corpus):
        """Returns lambdas satisfying algorithm in 5.19 of textbook"""
        lambda1 = lambda2 = 0

        for precedent in self.bigram_counts:
            for token in self.bigram_counts[precedent]:
                try:
                    case1 = ((self.bigram_counts[precedent][token] - 1) / 
                             (self.unigram_counts[precedent] - 1))
                except ZeroDivisionError:
                    case1 = 0

                try:
                    case2 = ((self.unigram_counts[token] - 1 ) /
                             (self.unigrams_observed - 1))
                except ZeroDivisionError:
                    case2 = 0

                if case1 > case2:
                    lambda2 += self.bigram_counts[precedent][token]
                else:
                    lambda1 += self.bigram_counts[precedent][token]

        sum = lambda1 + lambda2
        return (lambda1 / sum, lambda2 / sum)

    def CheckDistribution(self):
        """
        Ensures that for every valid unigram in the model, unigram and bigram 
        probabilities sum to 1
        """

        # Interpolation requires checking every possible bigram, which
        # leads to quadratic runtime.
        # However, if the un-smoothed model's distributions are correct, 
        # interpolation's distributions will be too. This code chunk exists only
        # to demonstrate that the code for determining interpolated
        # probabilities is mathematically sound.

        if self.interpolation:
            for precedent in self.bigram_counts:
                if precedent == end_token:
                    continue
                sum = 0.0
                for token in self.bigram_counts:
                    sum += self.GetBigramProbability(precedent, token)
                assert sum > (1 - 1e-09) and sum < (1 + 1e-09)

        # Un-smoothed models and Laplace models are quicker to calculate
        else:
            for precedent in self.bigram_counts:
                if precedent == end_token:
                    continue
                sum = 0.0

                # For each token that follows the precedent, add its probability
                for token in self.bigram_counts[precedent]:
                    sum += self.GetBigramProbability(precedent, token)

                # If Laplace smoothing is active, we want to avoid summing over
                # every single bigram observed only once
                if self.laplace:
                    precedent_count = self.unigram_counts[precedent]

                    # Calculate the probability of this bigram seen once
                    laplace_probability = \
                        (1 / (precedent_count + len(self.unigram_counts)))

                    # Multiply that probability by the number of bigrams only 
                    # seen once
                    total_probability = \
                        laplace_probability * \
                            (len(self.unigram_counts) - \
                             len(self.bigram_counts[precedent]))

                    sum += total_probability

                assert sum > (1 - 1e-09) and sum < (1 + 1e-09)

        # Finish by checking that unigram probabilities sum to 1
        return self.CheckDistributionUnigrams()

    def CheckDistributionUnigrams(self):
        """Ensures probability of all unigrams in model sum to 1"""
        sum = 0.0
        for word in self.unigram_counts:
            sum += self.GetUnigramProbability(word)
        assert sum > (1 - 1e-09) and sum < (1 + 1e-09)

        return True

    def Perplexity(self, corpus):
        """Returns perplexity of model based on test corpus"""
        prob_sum = 0
        M = 0
        for sentence in corpus:
            for i in range(len(sentence)):
                try:
                    (precedent, token) = (sentence[i], sentence[i + 1])
                except IndexError:
                    continue

                prob = self.GetBigramProbability(precedent, token)
                if prob == 0:
                    return float('+inf')
                else:
                    prob_sum += log(prob)
                M += 1

        exponent = -prob_sum / M
        perplexity = exp(exponent)
        return perplexity