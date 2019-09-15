def parse2(url):
    # Sep 15, 2019, Yingqi Ding, HopHack Fall 2019
    from tempfile import mkstemp
    from shutil import move
    from os import fdopen, remove
    import os
    import pandas as pd

    #url = 'https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber='
    fh, abs_path = mkstemp()
    file_path = 'amazon_reviews.py'
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
              if 'myBaseUrl = https://' in line:
                  new_file.write('    myBaseUrl = ' + url)
              else:
                new_file.write(line)

    remove(file_path)
    move(abs_path, file_path)

    os.system("scrapy runspider amazon_reviews.py -o reviews.csv")
    data = pd.read_csv('reviews.csv')
    rating = []
    review = []
    for i in range(len(data['stars'])):
      if data['stars'][i][:3] != 'sta':
        rating.append(float(data['stars'][i][:3]))
        review.append(data['comment'][i])

    print(data)

    return (rating, review)

tup = parse2('https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=')
print(tup[0])
