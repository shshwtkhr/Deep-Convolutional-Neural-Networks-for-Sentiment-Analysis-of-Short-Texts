from __future__ import print_function
import numpy

def read(inp_file):
    fin = open(inp_file, 'r')
    lines = fin.readlines()
    
    y = [] 
    xchr = []
    xwrd = []
    kchr = 3 
    kwrd = 5
    wordcnt = 0 
    charcnt = 0
    wordsmap = {}
    charmap = {} 
    
    
     

    y = [] 
    xchr = []
    xwrd = []

    maxwordlen, maxsenlen, numsent = 0, 0, 21000

    for line in lines[:numsent]:
        words = line[:-1].split()
        tokens = words[1:]
        y.append(int(float(words[0])))
        maxsenlen = max(maxsenlen,len(tokens))
        for token in tokens:
            if token not in wordsmap:
                wordsmap[token] = wordcnt
                wordcnt += 1
                maxwordlen = max(maxwordlen,len(token))
            for i in xrange(len(token)):
                if token[i] not in charmap:
                    charmap[token[i]] = charcnt
                    charcnt += 1
    
    for line in lines[:numsent]:
        words = line[:-1].split()
        tokens = words[1:]
        wordmat = [0] * (maxsenlen+kwrd-1)
        charmat = numpy.zeros((maxsenlen+kwrd-1, maxwordlen+kchr-1))

        for i in xrange(len(tokens)):
            wordmat[(kwrd/2)+i] = wordsmap[tokens[i]]
            for j in xrange(len(tokens[i])):
                charmat[(kwrd/2)+i][(kchr/2)+j] = charmap[tokens[i][j]]
        xchr.append(charmat)
        xwrd.append(wordmat)
    maxwordlen += kchr-1
    maxsenlen += kwrd-1

    data = (numsent, charcnt, wordcnt, maxwordlen, maxsenlen,\
            kchr, kwrd, xchr, xwrd, y)
    return data

read("tweets_clean.txt") 

