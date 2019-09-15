# Spam Slayer

Spam Slayer is a webapp used to detect deceptions from Amazon products' reviews. Users can copy a link to the product and paste directly paste it to our page. Using our trained deep learning model, Spam Slayer will distinguish between true/fake reviews and show them an adjusted rating without fake reviews.

## Repo's Structure
The main program for this project is `./span_slayer_runner.py`, which also has a copy in the front-end folder `./Spam_Slayer/slayer/`

## Dataset
We used dataset found on [Kaggle](https://www.kaggle.com/lievgarcia/amazon-reviews), which is adapted from raw data provided by [Amazon](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). The data contains 21,000 reviews, 10,500 labeled as truthful and 10,500 labeled as deceptive.

## Training
We have our own word-embedding generators, using vocab from the abovementioned dataset. Using two convolution layers, one for sentence convolution and another for document convolution, a maxpooling layer and a softmax layer for classification, the final validation accuracy is ~66%. Because of the limit of time, we don't have the chance to improve our architecture, but we got a significant 26% enhancement compared to human's success in distinguishing deceptive reviews (~40%).

## Front End
We built a web application with python and Django framework.
Our app crawls data from a given url (currently it works for an url from Amazon, but similar websites like Yelp can easily be done), and turn a list of reviews to our trained model to classify whether each review as real or computer-generated. They are then displayed on the UI.

To use the app on localhost:

pip install django

python manage.py runserver


Our model is based on research by Yingqi Ding:
https://github.com/dyq0811/Opinion-Spam-Detection-BiRCNN
