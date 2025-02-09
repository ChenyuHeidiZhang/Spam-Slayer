from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from six.moves import cPickle as pickle

import io
import re
import matplotlib.pyplot as plt
import gensim
import scipy.stats as stats


from django.shortcuts import render

ROOT = '/Users/chenyuzhang/desktop'

class SCNN_MODEL(object):
    '''
        A SCNN model for Deceptive spam reviews detection. 
        Use google word2vec.
    '''
    
    def __init__(self, sentence_per_review, words_per_sentence, wordVectors, embedding_size, 
                filter_widths_sent_conv, num_filters_sent_conv, filter_widths_doc_conv, num_filters_doc_conv, 
                num_classes, l2_reg_lambda=0.0,
                training=False):
        '''
        Attributes:
            sentence_per_review: The number of sentences per review
            words_per_sentence: The number or words per sentence
            wordVectors: The Word2Vec model
            embedding_size: the size of each word vector representation
            filter_widths_sent_conv: An array the contains the widths of the convolutional filters for the sentence convolution layer
            num_filters_sent_conv: the number of convolutional filters for the sentence convolution layer
            filter_widths_doc_conv: An array the contains the widths of the convolutional filters for the document convolution layer
            num_filters_doc_conv: the number of convolutional filters for the document convolution layer
            num_classes: The number of classes. 2 in this case.
            l2_reg_lambda: the lambda parameter for l2 regularization.
        '''
        
        #Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=(None, sentence_per_review * words_per_sentence), name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=(None, num_classes), name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.input_size = tf.placeholder(tf.int32, name='input_size')
        
        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)
        
        #Reshape the input_x to [input_size*sentence_per_review, words_per_sentence, embedding_size, 1]
        with tf.name_scope('Reshape_Input_X'):
            self.x_reshape = tf.reshape(self.input_x, [self.input_size*sentence_per_review, words_per_sentence])
            self.x_emb = tf.nn.embedding_lookup(wordVectors, self.x_reshape)
            shape = self.x_emb.get_shape().as_list()
            self.x_emb_reshape = tf.reshape(self.x_emb, [self.input_size*sentence_per_review, shape[1], shape[2], 1])
            #Cast self.x_emb_reshape from Float64 to Float32
            self.data = tf.cast(self.x_emb_reshape, tf.float32)
            
        # Create a convolution + maxpool layer + tanh activation for each filter size
        conv_outputs = []
        for i, filter_size in enumerate(filter_widths_sent_conv):
            with tf.name_scope('sent_conv-maxpool-tanh-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters_sent_conv]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_sent_conv]), name='b')
                conv = tf.nn.conv2d(
                    self.data,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.bias_add(conv, b)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, words_per_sentence - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                #Apply tanh Activation
                h_output = tf.nn.tanh(pooled, name='tanh')
                conv_outputs.append(h_output)
                
        # Combine all the outputs
        num_filters_total = num_filters_sent_conv * len(filter_widths_sent_conv)
        self.h_combine = tf.concat(conv_outputs, 3)
        self.h_combine_flat = tf.reshape(self.h_combine, [-1, num_filters_total])
        
        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_combine_flat, self.dropout)
        
        #Reshape self.h_drop for the input of the document convolution layer
        self.conv_doc_x = tf.reshape(self.h_drop, [self.input_size, sentence_per_review, num_filters_total])
        self.conv_doc_input = tf.reshape(self.conv_doc_x, [self.input_size, sentence_per_review, num_filters_total, 1])
        
        # Create a convolution + maxpool layer + tanh for each filter size
        conv_doc_outputs = []
        for i, filter_size in enumerate(filter_widths_doc_conv):
            with tf.name_scope('doc_conv-maxpool-tanh-%s' % filter_size):
                # Convolution Layer
                filter_shape_doc = [filter_size, num_filters_total, 1, num_filters_doc_conv]
                W_doc = tf.Variable(tf.truncated_normal(filter_shape_doc, stddev=0.1), name='W_doc')
                b_doc = tf.Variable(tf.constant(0.1, shape=[num_filters_doc_conv]), name='b_doc')
                conv_doc = tf.nn.conv2d(
                    self.conv_doc_input,
                    W_doc,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv_doc')
                h_doc = tf.nn.bias_add(conv_doc, b_doc)
                # Maxpooling over the outputs
                pooled_doc = tf.nn.max_pool(
                    h_doc,
                    ksize=[1, sentence_per_review - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool_doc')
                #Apply tanh Activation
                h_output_doc = tf.nn.tanh(pooled_doc, name='tanh')
                conv_doc_outputs.append(h_output_doc)
        
        # Combine all the outputs
        num_filters_total_doc = num_filters_doc_conv * len(filter_widths_doc_conv)
        self.h_combine_doc = tf.concat(conv_doc_outputs, 3)
        self.h_combine_flat_doc = tf.reshape(self.h_combine_doc, [-1, num_filters_total_doc])
        
        # Add dropout
        with tf.name_scope('dropout'):
            self.doc_rep = tf.nn.dropout(self.h_combine_flat_doc, self.dropout)
        
        #Softmax classification layers for final score and prediction
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total_doc, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.doc_rep, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
            
            
        if training:
            # Compute Mean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss      
                  
             # Compute Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

# Model Hyperparameters
SENTENCE_PER_REVIEW = 16
WORDS_PER_SENTENCE = 10
EMBEDDING_DIM = 300
FILTER_WIDTHS_SENT_CONV = np.array([3, 4, 5])
NUM_FILTERS_SENT_CONV = 100
FILTER_WIDTHS_DOC_CONV = np.array([3, 4, 5])
NUM_FILTERS_DOC_CONV = 100
NUM_CLASSES = 2
DROPOUT_KEEP_PROB = 0.5
L2_REG_LAMBDA = 0.0
BATCH_SIZE = 64
NUM_EPOCHS = 100
EVALUATE_EVERY = 100   # Evaluate model on the validation set after 100 steps
CHECKPOINT_EVERY = 100 # Save model after each 200 steps
NUM_CHECKPOINTS = 5    # Keep only the 5 most recents checkpoints
LEARNING_RATE = 1e-3   # The learning rate

# Load vocabulary and the word2vec model
pickle_file = ROOT+'/Spam-Slayer/Data/Kaggle Amazon Data/save.pickle'
with open(pickle_file, 'rb') as f :
    save = pickle.load(f)
    wordsVectors = save['wordsVectors']
    vocabulary = save['vocabulary']
    del save  # hint to help gc free up memory
# print('Vocabulary and the word2vec loaded')
# print('Vocabulary size is ', len(vocabulary))
# print('Word2Vec model shape is ', wordsVectors.shape)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def predict(x_batch):
    '''Making a prediction'''
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = SCNN_MODEL(sentence_per_review=SENTENCE_PER_REVIEW, 
                            words_per_sentence=WORDS_PER_SENTENCE, 
                            wordVectors=wordsVectors, 
                            embedding_size=EMBEDDING_DIM, 
                            filter_widths_sent_conv=FILTER_WIDTHS_SENT_CONV, 
                            num_filters_sent_conv=NUM_FILTERS_SENT_CONV, 
                            filter_widths_doc_conv=FILTER_WIDTHS_DOC_CONV, 
                            num_filters_doc_conv=NUM_FILTERS_DOC_CONV, 
                            num_classes=NUM_CLASSES, 
                            l2_reg_lambda=L2_REG_LAMBDA,
                            training=False)
            
            saver = tf.train.Saver()
            saver.restore(sess, ROOT+"/Spam-Slayer/Data/Kaggle Amazon Data/runs/model-14400")                     
            
            def get_logits_predictions(x_batch):
                """
                Evaluates model on input.
                """
                feed_dict = {cnn.input_x: x_batch,
                             cnn.input_size: len(x_batch),
                             cnn.dropout: 1.0}
                logits, prediction = sess.run([cnn.scores, cnn.predictions],
                                              feed_dict)               
                return logits, prediction

            return get_logits_predictions(x_batch)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def handle_reviews(single_review):
    cleanedLine = clean_str(single_review)
    cleanedLine = cleanedLine.strip()
    cleanedLine = cleanedLine.lower()
    words = cleanedLine.split(' ')
    return words, len(words)

MAX_SEQ_LENGTH = 160
def convert_string_to_index_array(single_review):

    doc = np.zeros(MAX_SEQ_LENGTH, dtype='int32')
    indexCounter = 0
    words,_ = handle_reviews(single_review)
    for word in words:
        try:
            doc[indexCounter] = vocabulary.index(word) # What if word is not found in vocabulary? 
        except:
            doc[indexCounter] = 0
        indexCounter = indexCounter + 1
        if (indexCounter >= MAX_SEQ_LENGTH):
            break
    return doc

def get_preprocessed_data(list_of_reviews):
    total_reviews = len(list_of_reviews)
    idsMatrix = np.ndarray(shape=(total_reviews, MAX_SEQ_LENGTH), dtype='int32')

    counter = 0
    for single_review in list_of_reviews:
        idsMatrix[counter] = convert_string_to_index_array(single_review)
        counter = counter + 1

    return idsMatrix

def real_time_predict_tester():
    # Test giving string inputs
    def get_list_of_string(path):
        list_of_reviews = []
        files = os.listdir(path)
        for name in files:
            full_path = os.path.join(path,name)
            # Open a file: file
            file = open(full_path,mode='r') 
            list_of_reviews.append(file.read())   
            # close the file
            file.close()
        return list_of_reviews

    root_path = ROOT+'/Spam-Slayer/Data/Gold Standard Data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold'
    list_of_reviews = []
    for i in range(1, 6):
        path = root_path + str(i)
        list_of_reviews = list_of_reviews + get_list_of_string(path)

    data = get_preprocessed_data(list_of_reviews)

    logits, prediction = predict(data)

    # print("logits:\n", logits)
    print("\n\npredictions:\n", prediction)

    l = len(prediction)
    right = l - np.sum(prediction)
    print("accuracy", right/l)

def real_time_predict(list_of_reviews):
    data = get_preprocessed_data(list_of_reviews)
    logits, prediction = predict(data)
    return prediction;


def parse2(url):
    # Sep 15, 2019, Yingqi Ding, HopHack Fall 2019
    from tempfile import mkstemp
    from shutil import move
    from os import fdopen, remove
    import os
    import pandas as pd

    #url = 'https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber='
    with open('/Users/chenyuzhang/desktop/Spam-Slayer/Spam_Slayer/slayer/amazon_reviews.py', 'w') as new_file:
        with open('/Users/chenyuzhang/desktop/Spam-Slayer/Spam_Slayer/slayer/amazon_reviews_old.py') as old_file:
            for n, line in enumerate(old_file):
              if n == 15:
                new_file.write(line[:line.index('=')+2] + '"' + url + '&pageNumber=' + '"')
              elif n >= 2:
                new_file.write(line)

    os.system("scrapy runspider /Users/chenyuzhang/desktop/Spam-Slayer/Spam_Slayer/slayer/amazon_reviews.py -o /Users/chenyuzhang/desktop/Spam-Slayer/Spam_Slayer/slayer/reviews.csv")
    data = pd.read_csv('/Users/chenyuzhang/desktop/Spam-Slayer/Spam_Slayer/slayer/reviews.csv')
    rating = []
    review = []
    for i in range(len(data['stars'])):
      if data['stars'][i][:3] != 'sta':
        rating.append(float(data['stars'][i][:3]))
        review.append(data['comment'][i])

    return (rating[:100], review[:100])


def parse(url):
    #import urllib.request
    import requests
    import urllib.parse
    import urllib.error
    from bs4 import BeautifulSoup
    import ssl
    import json

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # url=input("Enter Amazon Product Url- ")
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    #headers = {'User-Agent': 'Mozilla/5.0'}
    #page = requests.get(url, headers=headers)
    #soup = BeautifulSoup(page.text, 'html.parser')
    html = soup.prettify('utf-8')
    product_json = {}

    for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
        name_of_product = spans.text.strip()
        product_json['name'] = name_of_product
        break

    # for i_tags in soup.findAll('i',
    #                            attrs={'data-hook': 'average-star-rating'}):
    #     for spans in i_tags.findAll('span', attrs={'class': 'a-icon-alt'}):
    #         product_json['star-rating'] = spans.text.strip()
    #         break

    # for spans in soup.findAll('span', attrs={'id': 'acrCustomerReviewText'
    #                           }):
    #     if spans.text:
    #         review_count = spans.text.strip()
    #         product_json['customer-reviews-count'] = review_count
    #         break

    # product_json['short-reviews'] = []
    # for a_tags in soup.findAll('a',
    #                            attrs={'class': 'a-size-base a-link-normal review-title a-color-base a-text-bold'
    #                            }):
    #     short_review = a_tags.text.strip()
    #     product_json['short-reviews'].append(short_review)

    product_json['long-reviews'] = []
    for divs in soup.findAll('div', attrs={'data-hook': 'review-collapsed'}):
        long_review = divs.text.strip()
        product_json['long-reviews'].append(long_review)

    return product_json



def home(request):
    reviews_real = []
    reviews_fake = []
    product = "Amazon Product"

    count = 0
    rating_sum = 0
    count5star = 0
    count4star = 0
    count3star = 0
    count2star = 0
    count1star = 0

    rate = 0

    if (request.method == 'POST'): 
        url = request.POST.get('url-input', None)
        
        try:
             json = parse(url)
        except:
             json = parse('https://www.amazon.com/Doublju-Lightweight-Zip-Up-Hoodie-Jacket/dp/B01N67CJGX/ref=cm_cr_arp_d_product_top?ie=UTF8')

        # json = {
        #     "brand": "Bose",
        #     "name": "Bose QuietComfort 35 II Wireless Bluetooth Headphones, Noise-Cancelling, with Alexa voice control, enabled with Bose AR - Silver",
        #     "price": "$349.00",
        #     "img-url": "https://images-na.ssl-images-amazon.com/images/I/41I0FkzD56L._SY300_QL70_.jpg",
        #     "star-rating": "4.4 out of 5 stars",
        #     "customer-reviews-count": "3,411 customer reviews",
        #     "details": [
        #         "3 levels of world class noise cancellation for better listening experience in any environment",
        #         "Alexa enabled for voice access to music, information, and more",
        #         "Noise rejecting dual microphone system for clear sound and voice pick up",
        #         "Balanced audio performance at any volume",
        #         "Hassle free Bluetooth pairing, personalized settings, access to future updates, and more through the Bose Connect app",
        #         "Bose AR enabled : an innovative, audio only version of augmented reality",
        #         "Unlock Bose AR via a firmware update through the Bose Connect app",
        #         "Bose AR availability and functionality varies. Bose AR enhanced apps are currently available for iPhone and iPad users only. Apps for Android devices are in development"
        #     ],
        #     "short-reviews": [],
        #     "long-reviews": [
        #         "I give it five stars. My wife hates them she would give it 1. I put them on and that\u2019s it. I can\u2019t hear her any more.",
        #         "I purchased these and the Sony WH1000XM2 to compare the two. Cnet says they both have a \"9\" for sound quality. I would agree, they both sound excellent. The Bose won the test for its noise cancellation, performance talking to people on the phone, comfort on my head, and sound processing.Phone performance:I compared how the Bose and the Sony sounded when recording and playing back my voice with a fan running in the background. The Bose sounded like I was holding an old fashioned handset and talking in a quiet room - intimate and zero background noise. The Sony sounded like I was on speaker phone, and I could hear some background noise. (As a control I also recorded using neither and it sounded like I was on speaker but also I could hear more background noise.) This feature is important to me since I spend a lot of time on the phone and prefer my clients to not hear any background noise.Sound qualityThe Bose and Sony both have excellent sound quality for playing music. I personally prefer the sound the Sony produces. The Sony iphone app lets you choose your levels on an equalizer, and I like that. However, the Bose hears what type of music you're playing and automatically optimizes the sound, and it does a really good job. While I would prefer to be able to set the levels if I so choose, I also appreciate that Bose is making it all easy for me, so I can truly listen to my music on random and not have to fuss with levels. The Bose iphone app doesn't do very much at all. It does let you \"find your headset\" similar to the \"find my phone\" app, and it will apply firmware. (I'm hoping Bose will add an equalizer into its app in the future.)Noise cancellationThe Sony occasionally made me aware that noise cancellation was going on (with a whitenoise effect). The Bose on the other hand just stops the noise. There is no delay, no white noise, just quiet and your music.ControlsThe Bose controls are intuitive to find and to use. I like that the on-off control is a switch to flip on and off (rather than a button to find). Also, you can use this button to switch between devices, for example between your phone and ipad and your TV amplifier. The right earcup has three small buttons in a row together, and they control a lot of things. Volume, pause, and skip, rewind, answer/decline calls, etc. The left earcup only has the google assistant button, which I programmed to instead control the amount of noise cancellation (high, low, off). Song playback sounds much better with noise cancellation on high, and I don't think that has to do with noise (I was in a quiet environment); the bass sounds enhanced with noise cancellation on for some reason. (In comparison, the Sony lets you swipe the earcup itself to control volume, pause, play, skip, etc. This seems great in theory, but in practice if I bumped  the earcup adjusting my glasses or whatever, the music would pause. I found the Sony to be somewhat buggy in that regard. It would stop playing at times and I had to pick up my phone to get the music re-started, which is annoying.) I found the Bose controls to be more intuitive and consistent. Also, when you switch them on a voice tells you how much battery you have left, which is handy.ComfortThe Bose QuietComfort truly is comfortable. The earcups are soft, there is not a lot of clamping, and the top band is padded so it's less annoying on the top of the head.  (In comparison, the Sony do have more clamping which was uncomfortable over glasses.) I am a pilot and wear a similar headset, so I'm familiar with how headphones feel after a few hours. On-ear are not going to be as comfortable for long term wear as earbuds would be, but I was wiling to make that trade-off to get superior sound quality.StyleThe Bose are more streamlined to my head. the Sony are bulky and look geeky.ConclusionEven though the Sony produces superior sound, the litany of other features (superior noise cancellation, intuitive controls, comfort, style, & phone performance) won me over to the Bose.",
        #         "These are amazing and Bose is great!  I purchased these headphones for their noise-cancellation abilities.  I am an application developer and wear them at work because I am easily distracted.  I don't even listen to music with them... they just cancel just about all background noise, including random chatter from the annoying marketing department.  I also use them for WebEx and Skype calls with clients.  The built-in microphone is great and the noise-cancellation is helpful during those calls as well.  Battery life is extremely good.  I can go an entire week without having to charge the headset.  It comes with an 1/8\" cable that will allow you to listen to music even when the batteries are dead.  An added feature is that the bluetooth feature will allow you to connect to and hear audio from two sources at the same time.  This is great if you want to connect to your iPhone and laptop at the same time.After a month of use, the headphones button would no longer work to pair to a new device.  It would still power on and off, just the pairing didn't function.  Bose support was difficult to deal with.  At first the support technician didn't think there was anything wrong with the headset.  After a frustrating conversation, he relented and sent me instructions for sending them back to the manufacturer.  This would mean that I would be without my precious headphones for at least two weeks.  The next day prior to dropping the headphones at UPS I came across The Bose Store at Tysons Corner Mall.  I had the headphones with me so I decided to speak to one of the employees there.  After a very brief conversation, and even though I purchased these through Amazon, he walks over to a display, gets me a brand new set of headphones and proceeds to exchange my defective ones for the new set.  No other questions asked.  He just said \"We like to take care of our customers\".  I wish customer support would have been as easy to deal with... but, things worked out.",
        #         "I received these with firmware 2.0.1 which worked phenomenally with my Mac and iPhone. ANC was nice and bluetooth stable. Unfortunately, the google assistant app decided to update my firmware to 2.5.1 by it self!! Seriously Google? Ask me for permission first.Afterwards, ANC got noticeably worse. I played various background noise on my speakers and there was no difference between low and high ANC. Bluetooth skipped every 2 seconds if both Mac and iPhone were connected whereas it worked seamlessly before. Symptoms didn't go away after updating to the 3.1.8 firmware via Bose's website. These are getting returned.If you get these headphones, DO NOT update them - please. If it ain't broke don't fix it.Notes:-I did the Bose reset and re-paired multiple times. No dice.-This ANC issue happened on the original QC 35 too-google assistant initiated update WITHOUT prompting \"yes or no\". This happened when opening the app."
        #     ]
        # }
        try:
            tup = parse2(url)
        except:
            tup = ([],[])

        for rating,review in zip(tup[0],tup[1]):
        #for review in json["long-reviews"]:
            #rating = 3
            if (real_time_predict([review])[0]==0):
                reviews_real.append({
                        'author': 'CoreyMS',
                        'content': review,
                        'date_posted': 'August 27, 2018',
                        'rating': rating
                    })
                rating_sum = rating_sum + rating
                count = count + 1
                if rating==1:
                    count1star = count1star + 1
                if rating==2:
                    count2star = count2star + 1
                if rating==3:
                    count3star = count3star + 1
                if rating==4:
                    count4star = count4star + 1
                if rating==5:
                    count5star = count5star + 1
            else:
                reviews_fake.append({
                        'author': 'CoreyMS',
                        'content': review,
                        'date_posted': 'August 27, 2018',
                        'rating': rating
                    })
        if ('name' in json):
            product = json['name']
    
    percs = [0,0,0,0,0]
    if count != 0:
        rate = round(rating_sum/count, 2)
        percs[0] = round(count1star/count, 2)
        percs[1] = round(count2star/count, 2)
        percs[2] = round(count3star/count, 2)
        percs[3] = round(count4star/count, 2)
        percs[4] = round(count5star/count, 2)

    context = {
        'reviews_real': reviews_real,
        'reviews_fake': reviews_fake,
        'product': product,
        'rating': rate,
        'count': count,
        'count5star': count5star,
        'count4star': count4star,
        'count3star': count3star,
        'count2star': count2star,
        'count1star': count1star,
        'percs': percs,
    }
    return render(request, 'slayer/home.html', context)


def yelp(request):
    reviews = []
    context = {
        'reviews': reviews
    }
    return render(request, 'slayer/yelp.html', context)
