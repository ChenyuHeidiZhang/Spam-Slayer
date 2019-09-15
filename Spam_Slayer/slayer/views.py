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
pickle_file = '/Users/chenyuzhang/desktop/Spam-Slayer/Data/save.pickle'
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

def train():
    '''Training the model'''

    #Load training data, training labels, validation data, validation labels
    pickle_file = '/Users/chenyuzhang/desktop/Spam-Slayer/Data/data_saved.pickle'
    with open(pickle_file, 'rb') as f :
        save = pickle.load(f)
        train_data = save['train_data']
        train_labels = save['train_labels']
        validation_data = save['validation_data']
        validation_labels = save['validation_labels']
        del save  # hint to help gc free up memory
    print('train data shape ', train_data.shape)
    print('train labels shape ', train_labels.shape)
    print('validation data shape ', validation_data.shape)
    print('validation labels shape ', validation_labels.shape)

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
                            training=True)
            
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join("/Users/chenyuzhang/desktop/Spam-Slayer/Data/runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.input_size: len(y_batch),
                    cnn.dropout: DROPOUT_KEEP_PROB
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
            
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.input_size: y_batch.shape[0],
                    cnn.dropout: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            
            print('train data shape ', train_data.shape)
            print('train labels shape ', train_labels.shape)
            print('validation data shape ', validation_data.shape)
            print('validation labels shape ', validation_labels.shape)
            
            # Generate batches
            batches = batch_iter(
                list(zip(train_data, train_labels)), BATCH_SIZE, NUM_EPOCHS)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % EVALUATE_EVERY == 0:
                    print("\nEvaluation:")
                    dev_step(validation_data, validation_labels, writer=dev_summary_writer)
                    print("")
                if current_step % CHECKPOINT_EVERY == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


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
            saver.restore(sess, "/Users/chenyuzhang/desktop/Spam-Slayer/Data/runs/model-900")                     
            
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

    root_path = '/Users/chenyuzhang/desktop/Spam-Slayer/Data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold'
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
    #html = urllib.request.urlopen(url, context=ctx).read()
    headers = {'User-Agent': 'Mozilla/5.0'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    html = soup.prettify('utf-8')
    product_json = {}

    for divs in soup.findAll('div', attrs={'class': 'a-box-group'}):
        try:
            product_json['brand'] = divs['data-brand']
            break
        except:
            pass

    for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
        name_of_product = spans.text.strip()
        product_json['name'] = name_of_product
        break

    for divs in soup.findAll('div'):
        try:
            price = str(divs['data-asin-price'])
            product_json['price'] = '$' + price
            break
        except:
            pass

    for divs in soup.findAll('div', attrs={'id': 'rwImages_hidden'}):
        for img_tag in divs.findAll('img', attrs={'style': 'display:none;'}):
            product_json['img-url'] = img_tag['src']
            break

    for i_tags in soup.findAll('i',
                               attrs={'data-hook': 'average-star-rating'}):
        for spans in i_tags.findAll('span', attrs={'class': 'a-icon-alt'}):
            product_json['star-rating'] = spans.text.strip()
            break

    for spans in soup.findAll('span', attrs={'id': 'acrCustomerReviewText'
                              }):
        if spans.text:
            review_count = spans.text.strip()
            product_json['customer-reviews-count'] = review_count
            break

    product_json['details'] = []
    for ul_tags in soup.findAll('ul',
                                attrs={'class': 'a-unordered-list a-vertical a-spacing-none'
                                }):
        for li_tags in ul_tags.findAll('li'):
            for spans in li_tags.findAll('span',
                    attrs={'class': 'a-list-item'}, text=True,
                    recursive=False):
                product_json['details'].append(spans.text.strip())

    product_json['short-reviews'] = []
    for a_tags in soup.findAll('a',
                               attrs={'class': 'a-size-base a-link-normal review-title a-color-base a-text-bold'
                               }):
        short_review = a_tags.text.strip()
        product_json['short-reviews'].append(short_review)


    product_json['long-reviews'] = []
    for divs in soup.findAll('div', attrs={'data-hook': 'review-collapsed'}):
        long_review = divs.text.strip()
        product_json['long-reviews'].append(long_review)

    return product_json


def home(request):
    reviews_real = []
    reviews_fake = []
    product = "Amazon Product"

    count = 1
    rating_sum = 0
    count5star = 0
    count4star = 0
    count3star = 0
    count2star = 0
    count1star = 0

    if (request.method == 'POST'): 
        url = request.POST.get('url-input', None)
        try:
            json = parse(url)
        except:
            json = parse('https://www.amazon.com/Doublju-Lightweight-Zip-Up-Hoodie-Jacket/dp/B01N67CJGX/ref=cm_cr_arp_d_product_top?ie=UTF8')
        for review in json["long-reviews"]:
            rating = 4
            if (real_time_predict([review])[0]==0):
                reviews_real.append({
                        'author': 'CoreyMS',
                        'title': 'Review Title',
                        'content': review,
                        'date_posted': 'August 27, 2018',
                        'rating': rating
                    })
                rating_sum = rating_sum + 4
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
                        'title': 'Review Title',
                        'content': review,
                        'date_posted': 'August 27, 2018',
                        'rating': rating
                    })

        #product = json['name']

    context = {
        'reviews_real': reviews_real,
        'reviews_fake': reviews_fake,
        'product': product,
        'rating': rating_sum/count,
        'count': count,
        'count5star': count5star,
        'count4star': count4star,
        'count3star': count3star,
        'count2star': count2star,
        'count1star': count1star
    }
    return render(request, 'slayer/home.html', context)


def yelp(request):
    reviews = []
    context = {
        'reviews': reviews
    }
    return render(request, 'slayer/yelp.html', context)
