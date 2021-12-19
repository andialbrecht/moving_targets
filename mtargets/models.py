from datetime import timedelta

import numpy as np

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.utils import timezone

from mtargets import renderers


class Project(models.Model):

    name = models.CharField(max_length=200)
    renderer = models.CharField(max_length=200)
    generations = models.IntegerField(default=0)
    last_evolved = models.DateTimeField(blank=True, null=True)

    op_bits = 5
    param_bits = 12
    param_max_value = 2**param_bits - 1
    gene_bits = op_bits + 4 * param_bits
    initial_sequence_length = 7

    crossover_max = 10
    seq_min_length = 2
    seq_max_length = 120
    mutation_rate = 0.2
    pop_min = 10
    pop_max = 100
    initial_fitness = 0.5

    def __str__(self):
        return self.name

    def get_fittest(self):
        query = self.population.filter(died_at__isnull=True)
        query = query.order_by('-fitness')
        return query

    def recalc_fitness(self):
        for item in self.population.filter(died_at__isnull=True):
            # Mark mostly white images as dead
            path = settings.RENDERINGS_DIR / f'{item.pk}.jpg'
            if not path.exists():
                item.died_at = timezone.now()
                item.fitness = 0
            else:
                # Make them loose fitness the older they get
                item.fitness = item.fitness * 0.98
            item.save()

    def reduce_population(self, num_new=0):
        query = self.population.filter(died_at__isnull=True)
        query = query.order_by('fitness')
        num_total = query.count()
        num_to_reduce = num_total + num_new - self.pop_max
        if num_to_reduce > 0:
            tbd = query[:num_to_reduce]
            for item in tbd:
                item.died_at = timezone.now()
                item.died_gen = self.generations
                item.save()

    def one_item(self):
        """Returns one item to present it to the user."""
        candidates = []
        query = self.population.all()
        query = query.filter(died_at__isnull=True)
        # Avoid to show images twice
        delta = timezone.now() - timedelta(minutes=30)
        query = query.filter(
            Q(shown_at__lt=delta) | Q(shown_at__isnull=True))
        # Add 10 of 20 fittest
        query_fittest = query.filter(fitness__gt=self.initial_fitness)
        query_fittest = query_fittest.order_by('-fitness')
        fittest = list(query_fittest[:20])
        candidates.extend(list(
            np.random.choice(fittest, size=min(10, len(fittest)))))
        # Add 5 of 15 new ones
        query_newbies = query.filter(fitness=self.initial_fitness)
        query_newbies = query_newbies.order_by('-born_at')
        newbies = list(query_newbies[:15])
        candidates.extend(list(
            np.random.choice(newbies, size=min(5, len(newbies)))))
        # Pick 2 less popular
        query_unfit = query.filter(fitness__lt=self.initial_fitness)
        query_unfit = query_unfit.order_by('-fitness')
        unfit = list(query_unfit[:20])
        candidates.extend(list(
            np.random.choice(unfit, size=min(2, len(unfit)))))
        if not candidates:
            candidates = list(
                self.population.filter(died_at__isnull=True))
        item = np.random.choice(candidates)
        item.shown_at = timezone.now()
        item.save()
        return item


class Genotype(models.Model):

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='population')
    sequence = models.JSONField()
    parent1 = models.ForeignKey(
        'Genotype', blank=True, null=True, on_delete=models.SET_NULL,
        related_name='child1')
    parent2 = models.ForeignKey(
        'Genotype', blank=True, null=True, on_delete=models.SET_NULL,
        related_name='child2')
    born_at = models.DateTimeField(auto_now_add=True)
    born_gen = models.IntegerField(blank=True, null=True)
    died_at = models.DateTimeField(blank=True, null=True)
    died_gen = models.IntegerField(blank=True, null=True)
    shown_at = models.DateTimeField(blank=True, null=True)
    fitness = models.FloatField(default=0.5)
    last_voted = models.DateTimeField(blank=True, null=True)
    pde = models.TextField(blank=True, null=True)

    @property
    def sequence_length(self):
        return len(self.sequence)

    def render(self):
        r = renderers.get_renderer(self.project)
        self.pde = r.render_pde(self.sequence)
        if not settings.RENDERINGS_DIR.exists():
            settings.RENDERINGS_DIR.mkdir()
        with open(settings.RENDERINGS_DIR / f'{self.pk}.jpg', 'wb') as f:
            data, is_dead = r.render_image(self.pde)
            f.write(data)
            if is_dead:
                self.died_at = timezone.now()
                self.fitness = 0

    def render_stream(self, width=None, height=None):
        r = renderers.get_renderer(self.project)
        if width:
            r.w = width
        if height:
            r.h = height
        self.pde = r.render_pde(self.sequence)
        return r.render_image(self.pde)[0]

    def image_url(self):
        return f'renderings/{self.pk}.jpg'

    def image_path(self):
        return settings.RENDERINGS_DIR / f'{self.pk}.jpg'
