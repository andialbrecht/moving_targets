# Generated by Django 3.1.5 on 2021-01-26 18:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mtargets', '0005_genotype_shown_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='genotype',
            name='parent1',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='child1', to='mtargets.genotype'),
        ),
        migrations.AddField(
            model_name='genotype',
            name='parent2',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='child2', to='mtargets.genotype'),
        ),
    ]