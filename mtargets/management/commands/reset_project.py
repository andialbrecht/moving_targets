from django.core.management.base import BaseCommand

from mtargets import models


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('project_id', nargs=1, type=int)

    def handle(self, *args, **options):
        p = models.Project.objects.get(pk=options['project_id'][0])
        for item in p.population.all():
            if item.image_path().exists():
                item.image_path().unlink()
            item.delete()
        p.generations = 0
        p.save()
