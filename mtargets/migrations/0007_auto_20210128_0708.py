# Generated by Django 3.1.5 on 2021-01-28 07:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mtargets', '0006_auto_20210126_1846'),
    ]

    operations = [
        migrations.AddField(
            model_name='genotype',
            name='last_voted',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='last_evolved',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
