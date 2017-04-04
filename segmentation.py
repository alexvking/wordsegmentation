# Alex King
# COMP 150NLP
# Final Project

# Definition of the Segmenter class, which holds a language model inside it and
# performs segmentation and testing as necessary

import sys
from languagemodel import BigramLM
from math import log, exp
from nltk.tokenize import word_tokenize

start_token = "<S>"
end_token = "</S>"

class Segmenter():
    """ A class to segment text, holding a probabilistic language inside it"""
    def __init__(self, training_data):
        training = training_data
        self._vocabulary = self.BuildVocabulary(training)
        training_prep = self.PreprocessText(training, self._vocabulary)
        self._model = BigramLM(self._vocabulary)
        self._model.EstimateBigrams(training_prep)
        self._model.LaplaceSmooth(True)

    def BuildVocabulary(self, list_of_sentences):
        """Returns occurrence dictionary of all words in list of sentences"""
        occurrences = {}
        for sentence in list_of_sentences:
            for word in sentence:
                if occurrences.has_key(word):
                    occurrences[word] += 1
                else:
                    occurrences[word] = 1
        return occurrences

    def PreprocessText(self, list_of_sentences, vocabulary):
        """
        Preprocesses list of sentences by designating sentence boundaries and
        unknown words
        """
        # Iterate through original sentences, modifying them according to rules
        output_sentences = []
        for sentence in list_of_sentences:
            output_sentence = []
            output_sentence.append(start_token)
            for word in sentence:
                if word in vocabulary and vocabulary[word] > 0:
                    output_sentence.append(word)
                else:
                    output_sentence.append(unknown_token)
            output_sentence.append(end_token)
            output_sentences.append(output_sentence)

        return output_sentences

    def Segment(self, word):
        """
        Uses quality function to efficiently compute best segmentation of 
        string using dynamic programming
        """
        sentence_length = len(word)

        # Initialize table to hold probabilities
        table = [0 for i in xrange(sentence_length)] 

        for row in range(sentence_length - 1, -1, -1):

            segment = word[row:sentence_length]

            segmentations = [[segment[:i+1]] + 
                              table[i + row + 1] 
                              for i in range(len(segment) - 1)]
            segmentations.append([segment])

            scores = [self.Quality(sent) 
                      for sent in segmentations]

            table[row] = segmentations[scores.index(max(scores))]

        return table[0]

    def Maxmatch(self, str):
        """
        Segments sentence each time a word is seen, starting at the end;
        leads to longest word being created at each step
        (Baseline algorithm with moderately strong performance)
        """
        sent = []
        while len(str) > 0:
            found = False
            for i in range(len(str), -1, -1):
                if str[:i] in self._model.unigram_counts:
                    sent.append(str[:i])
                    str = str[i:]
                    found = True
                    break

            if not found:
                sent.append(str[0])
                str = str[1:]

        return sent

    def Minmatch(self, str):
        """
        Segments sentence each time a word is seen, starting at the beginning;
        leads to shortest word being created at each step
        (Alternative baseline algorithm with poor performance)
        """
        sent = []
        while len(str) > 0:
            found = False
            for i in range(len(str)):
                if str[:i] in self._model.unigram_counts:
                    sent.append(str[:i])
                    str = str[i:]
                    found = True
                    break

            if not found:
                sent.append(str[0])
                str = str[1:]

        return sent

    def Accuracy(self, ground_truth):
        """Returns the fraction of correctly segmented sentences in guess"""
        predicted = []
        precisions = 0
        recalls = 0
        for i in range(len(ground_truth)):
            # This function call can be changed to maxmatch to test its accuracy
            sent = self.Segment(''.join(ground_truth[i]))
            predicted.append(sent)

        num_correct = 0
        for i in range(len(ground_truth)):

            if predicted[i] == ground_truth[i]:
                num_correct += 1

            else:
                print "Predicted:", predicted[i]
                print "Actual:", ground_truth[i]
                print

            (precision, recall) = WordsCorrect(predicted[i], ground_truth[i])
            precisions += precision
            recalls += recall

        print "Average P:", float(precisions) / len(ground_truth)
        print "Average R:", float(recalls) / len(ground_truth)

        return float(num_correct) / len(ground_truth)

    def Quality(self, sent):
        """
        Returns chained probability of sentence based on unigram
        Laplace-smoothed model
        """
        prob_sum = 0

        for i in range(len(sent)):

            prob = self._model.GetUnigramProbability(sent[i])

            prob_sum += log(prob)

        return prob_sum

def WordsCorrect(predicted, truth):
    """
    Given two sentences, finds which words were correctly segmented and
    returns precision and recall
    """
    iter_predicted  = len(predicted[0])
    iter_truth      = len(truth[0])
    index_predicted = 1
    index_truth     = 1
    correct         = 0
    correct_words  = []

    predicted += ["x"]
    truth += ["x"]

    while index_predicted < len(predicted) and index_truth < len(truth):

        # Both at same breakpoint, so if the previous words were the same,
        # count them as correct, then grab the next word
        if iter_predicted == iter_truth:

            if predicted[index_predicted - 1] == truth[index_truth - 1]:
                correct += 1
                correct_words.append(predicted[index_predicted - 1])

            iter_predicted += len(predicted[index_predicted])
            index_predicted += 1

            iter_truth += len(truth[index_truth])
            index_truth += 1

        # We're further on predicted than truth, so truth needs to grab the 
        # next word
        elif iter_predicted > iter_truth:
            iter_truth += len(truth[index_truth])
            index_truth += 1

        elif iter_truth > iter_predicted:
            iter_predicted += len(predicted[index_predicted])
            index_predicted += 1

    # Account for the dummy item when using sentence length
    precision = float(correct) / (len(truth) - 1)
    recall    = float(correct) / (len(predicted) - 1)
    return (precision, recall)