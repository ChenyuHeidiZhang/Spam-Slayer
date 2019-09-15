# Spam Slayer

Spam Slayer is a webapp used to detect deceptions from Amazon products' reviews. Users can copy a link to the product and paste directly paste it to our page. Using our trained deep learning model, Spam Slayer will distinguish between true/fake reviews and show them an adjusted rating without fake reviews.

## Repo's Structure
The main program for this project is `./span_slayer_runner.py`, which also has a copy in the front-end folder `./Spam_Slayer/slayer/`

## Dataset
We used dataset found on [Kaggle](https://www.kaggle.com/lievgarcia/amazon-reviews), which is adapted from raw data provided by [Amazon](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). The data contains 21,000 reviews, 10,500 labeled as truthful and 10,500 labeled as deceptive.

## Training
We have our own word-embedding generators, using vocab from the abovementioned dataset. Using two convolution layers, one for sentence convolution and another for document convolution, a maxpooling layer and a softmax layer for classification, the final validation accuracy is ~66%. Because of the limit of time, we don't have the chance to improve our architecture, but we got a significant 26% enhancement compared to human's success in distinguishing deceptive reviews (~40%).

## Front End
We built a web application with python and Django framework, because we need to use machine learning and python is the easiest.
Our app crawls data from a given url (currently it works for an url from Amazon, but similar websites like Yelp can easily be done), and turn a list of reviews to our trained model.

























The reviews inputed to front-end should be in this format:

reviews = [
    {
        'author': 'CoreyMS',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'August 27, 2018',
        'label': 'Fake'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'August 28, 2018',
        'label': 'True'
    }
]

list of things needed to be crawled:
For each review:
author's profile picture?
author (link to author's Amazon profile)
rating the author gives
title of the review
date
content

Feed the review into the model and obtain:
label (real or fake)
