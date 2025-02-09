# Generated by Django 2.2.5 on 2019-09-14 19:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Review',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('author', models.CharField(max_length=100)),
                ('rating', models.CharField(default='5', max_length=10)),
                ('title', models.CharField(max_length=100)),
                ('date_posted', models.DateTimeField()),
                ('content', models.TextField()),
                ('label', models.IntegerField(default=1)),
            ],
        ),
    ]
