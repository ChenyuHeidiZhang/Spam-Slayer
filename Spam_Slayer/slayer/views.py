from django.shortcuts import render

def parse(url):
    import urllib.request
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
    reviews = [
        {
            'author': 'DEMO',
            'title': 'DEMO',
            'content': 'DEMO',
            'date_posted': 'DEMO'
        }
    ]
    product = "Amazon Product"
    if (request.method == 'POST'): 
        url = request.POST.get('url-input', None)
        try:
            json = parse(url)
        except:
            json = parse('https://www.amazon.com/Doublju-Lightweight-Zip-Up-Hoodie-Jacket/dp/B01N67CJGX/ref=cm_cr_arp_d_product_top?ie=UTF8')
        reviews = []
        for review in json["long-reviews"]:
            reviews.append({
                    'author': 'CoreyMS',
                    'title': 'Review Title',
                    'content': review,
                    'date_posted': 'August 27, 2018'
                })
        product = json['name']

    context = {
        'reviews': reviews,
        'product': product
    }
    return render(request, 'slayer/home.html', context)


def yelp(request):
    reviews = []
    context = {
        'reviews': reviews
    }
    return render(request, 'slayer/yelp.html', context)
