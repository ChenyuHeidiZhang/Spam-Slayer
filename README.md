# Spam-Slayer

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
