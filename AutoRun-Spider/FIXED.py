# Sep 15, 2019, Yingqi Ding, HopHack Fall 2019
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import os
import pandas as pd

url = 'https://www.amazon.com/Headphones-QuietComfort-SoundLink-Protective-Accessories/product-reviews/B06Y4TWNK9/ref=cm_cr_dp_d_show_all_top?ie=UTF8&reviewerType=all_reviews'
with open('amazon_reviews.py', 'w') as new_file:
    with open('amazon_reviews_old.py') as old_file:
        for n, line in enumerate(old_file):
          if n == 15:
            new_file.write(line[:line.index('=')+2] + '"' + url + '&pageNumber=' + '"')
          elif n >= 2:
            new_file.write(line)
os.system("scrapy runspider amazon_reviews.py -o reviews.csv")
data = pd.read_csv('reviews.csv')
rating = []
review = []
for i in range(len(data['stars'])):
  if data['stars'][i][:3] != 'sta':
    rating.append(float(data['stars'][i][:3]))
    review.append(data['comment'][i])