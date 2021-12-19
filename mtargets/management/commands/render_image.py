import argparse

from django.core.management.base import BaseCommand

from mtargets import models


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('genotype_id', nargs=1, type=int)
        parser.add_argument('output', nargs=1, type=argparse.FileType('wb'))
        parser.add_argument('--size', default=None)

    def handle(self, *args, **options):
        g = models.Genotype.objects.get(pk=options['genotype_id'][0])
        if options['size']:
            w, h = map(int, options['size'].split('x'))
        else:
            w, h = None, None
        f = options['output'][0]
        f.write(g.render_stream(w, h))
        f.close()
