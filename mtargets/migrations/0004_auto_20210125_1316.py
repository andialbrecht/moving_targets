# Generated by Django 3.1.5 on 2021-01-25 13:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mtargets', '0003_auto_20210125_1314'),
    ]

    operations = [
        migrations.AddField(
            model_name='genotype',
            name='born_gen',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='genotype',
            name='died_gen',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
