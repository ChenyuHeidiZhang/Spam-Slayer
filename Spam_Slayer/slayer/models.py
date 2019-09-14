from django.db import models

reviews = [
    {
        'author': 'CoreyMS',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'August 27, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'August 28, 2018'
    }
]

class Review(models.Model):
	author = models.CharField(max_length=100)
	rating = models.CharField(max_length=10, default = '5')
	title = models.CharField(max_length=100)
	date_posted = models.DateTimeField()
	content = models.TextField()
	label = models.IntegerField(default=1)

	def __str__(self):
		return self.title
