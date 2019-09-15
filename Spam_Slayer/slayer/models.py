from django.db import models

class Review(models.Model):
	author = models.CharField(max_length=100)
	rating = models.CharField(max_length=10, default = '5')
	title = models.CharField(max_length=100)
	date_posted = models.DateTimeField()
	content = models.TextField()
	label = models.IntegerField(default=1)

	def __str__(self):
		return self.title
