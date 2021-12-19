from django.core.management.base import BaseCommand

from mtargets import models


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('genotype_id', nargs=1, type=int)

    def handle(self, *args, **options):
        g = models.Genotype.objects.get(pk=options['genotype_id'][0])
        for line in g.pde.splitlines():
            if line.strip().startswith('save('):
                line = f'// {line}'
            elif line.strip().startswith('exit('):
                line = f'// {line}'
            print(line)
