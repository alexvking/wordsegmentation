# Alex King
# COMP 150NLP
# Final Project

# Driver for segmentation project

from segmentation import Segmenter
from nltk.tokenize import word_tokenize
import sys
import random
import time

def main():
    if len(sys.argv) != 2:
        print "Usage: python main.py [english | french | turkish]"
        exit(1)

    if sys.argv[1] == "english":
        file = "/data/aking09_nlp/en_news_300k.txt"
    elif sys.argv[1] == "french":
        file = "/data/aking09_nlp/fr_wiki_300k.txt"
    elif sys.argv[1] == "turkish":
        file = "/data/aking09_nlp/tur_mixed_300k.txt"
    else:
        print ("Unrecognized language. " + 
               "Choose from 'english', 'french', or 'turkish'")
        exit(1)

    print "Reading in corpus..."
    with open(file, "r") as f:
        content = f.readlines()

    i = 0
    sents = []

    # Seed random the same way each time for reproducible orderings
    random.seed(12345)
    r = list(range(len(content)))
    random.shuffle(r)

    # Shuffle the alphabetized corpus
    for i in r:

        sentence = content[i].decode('utf-8').split("\t")

        tokenized = word_tokenize(sentence[1])

        # Accounting for odd formatting of corpus
        if tokenized[-1] == u"''":
            tokenized.pop()

        sents.append(tokenized)

        i += 1
        if i == 295000:
            break

    print "Initializing Segmenter. This may take a couple of minutes..."
    s = Segmenter(sents[:290000])

    # Uncomment the below two lines to briefly test the segmentation accuracy,
    # precision and recall

    test = sents[-100:]
    print "Accuracy:", s.Accuracy(test)

    query = raw_input("Enter a string with no spaces to have segmented," + 
                      " or 'quit' to quit:\n").decode("utf-8")
    while query != 'quit':
        segmentation = s.Segment(query)
        for word in segmentation:
            sys.stdout.write(word + " ")
        sys.stdout.write("\n")
        query = raw_input("Enter a string with no spaces to have segmented," + 
                      " or 'quit' to quit:\n").decode("utf-8")

    return

if __name__ == '__main__':
    main()
