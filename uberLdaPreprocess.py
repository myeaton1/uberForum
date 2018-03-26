import os, string, re, pickle, nltk, gensim, json
import pandas as pd
import numpy as np
from itertools import compress
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger
from gensim import corpora
from collections import defaultdict

# Path to the data folder
datapath = '/Users/mattbook/Documents/Research/uber/data/'

# if first run, set all to true
do_setup = False
preprocess = False
doLda = True

if do_setup:

    print('Step 1: Importing data...')
    # Define exclude_set
    exclude_set = {}
    for ch in string.punctuation:
        exclude_set[ord(ch)] = ' '
    for x in range(10):
        exclude_set[ord(str(x))] = ' '

    all_posts = []
    all_filenames = []
    count = 0
    # Walk through all subdirectories in datapath
    for root, _, filenames in os.walk(datapath):
        # For each file
        for filename in filenames:
            # Check whether it is a valid file
            if not filename.startswith('.'):
                # try:
                # import file and process json
                with open(os.path.join(root, filename),'r') as inFile:
                    comments = json.load(inFile)

                indPosts = [comments['posts'][com]['content'] for com in comments['posts']]
                for body in indPosts:
                    body = body.lower()
                    body = body.translate(exclude_set)
                    all_posts.append(' '.join(body.split()))

    # this step is not necessary since just posts for now, can add
    # dates, etc. down the road
    print('Step 2: Cleaning up work space and transforming...')
    # all_data = {'posts':np.array(all_dates),'posts':all_posts}

    # drop the list versions
    all_filenames = None

    print('Step 3: Pickling...')
    with open('uberPostData.pickle','wb') as f:
        pickle.dump(all_posts,f)

else:
    posts = pickle.load(open('uberPostData.pickle','rb'))

if preprocess:
    # remove common words and tokenize
    print('Step 1: Handling stop words and tokenizing...')
    stopwords_list = nltk.corpus.stopwords.words('english')
    # Create a list for the tokenized sentences:
    tok_sentences = list()
    # Create a list for the tokenized reviews:
    tok_reviews = list()
    # Create a list for the sentence assigned POS tags:
    pos_sentences = list()
    # Create a translation table for removing the punctuation marks:
    translator = str.maketrans('', '', string.punctuation)

    print('  Tokenization progress:')
    all_words = list()
    r_count = 0
    nPosts = len(posts)
    # printFlag = True
    for post in posts:
        r_count += 1
        pctDone = round((r_count/nPosts),2)
        # if (pctDone*100 % 10 == 0) and printFlag = True:
            # printFlag = False
            # print('    {:.2%}'.format(pctDone))

        sentences = nltk.sent_tokenize(post)
        review_words= list()
        for sentence in sentences:
            sent_words = nltk.word_tokenize(sentence)
            sent_words_tok = [word.lower() for word in sent_words if word not in stopwords_list and word.isalpha() and len(word) > 1]
            tok_sentences.append(sent_words_tok)
            for words in sent_words_tok:
                all_words.append(words)
                review_words.append(words)
        tok_reviews.append(review_words)

    # create freq counts
    print('Step 2: Creating frequency counts...')
    frequency_count = nltk.FreqDist(all_words)
    words =np.array([word for word in frequency_count.keys()])
    word_freq=np.array([word for word in frequency_count.values()])
    freq_sort = np.argsort(word_freq)[::-1]
    word_freq_sort =word_freq[freq_sort]
    words_sorted = words[freq_sort]

    # create effective vocabulary removing words that appear only once
    # or are the top 25 words, this is adhoc, should change
    print('Step 3: Creating effective vocabulary...')
    rank=1
    effective_vocab=list()
    for object in words_sorted:
        if (rank>=28):
            fc = frequency_count[object]
            if (fc>1):
                effective_vocab.append(object)
        rank+=1

    print('Step 4: Removing words outside vocabulary...')
    tok_reviews_ev = list()
    effective_vocab_set = set(effective_vocab)
    countVar = 0
    nPosts = len(tok_reviews)
    for review in tok_reviews:
        countVar += 1
        review = set(review)
        # review_words_ev = [word for word in review if word in effective_vocab_set]
        review_words_ev = review.intersection(effective_vocab_set)
        tok_reviews_ev.append(list(review_words_ev))
        completePercent = (countVar / nPosts)*100
        if completePercent % 5 == 0:
            print('  %.0f%% complete' % completePercent)

    print('Step 5: Creating dictionary...')
    dictionary = corpora.Dictionary(tok_reviews_ev)
    dictionary.save('uberPostDict.dict')
    print('Step 6: Creating corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in tok_reviews_ev]

    # pickle the important variables
    print('Step 7: Pickling relevant variables...')
    with open('uberPostCorpus.pickle','wb') as f:
        pickle.dump(corpus,f)
    with open('uberPostDict.pickle','wb') as f:
        pickle.dump(dictionary,f)
    with open('uberPostSorted.pickle','wb') as f:
        pickle.dump(posts,f)

else:
    dictionary = Dictionary.load('uberPostDict.dict')
    corpus = pickle.load(open('uberPostCorpus.pickle','rb'))

if doLda:
    print('LDA TIME...')

    # number of topics currently ad hoc
    nTopics = 20

    # 50 iterations not enough
    lda = gensim.models.ldamulticore.LdaMulticore(corpus,id2word=dictionary, num_topics=nTopics, iterations=50, alpha='symmetric')

    # pickle the output
    with open('uberPostLda_'+str(nTopics)+'.pickle','wb') as f:
        pickle.dump(lda,f)

else:
    lda = pickle.load(open('uberPostLda.pickle','rb'))


