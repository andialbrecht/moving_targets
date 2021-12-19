from django.core.management.base import BaseCommand

from mtargets.evolver import Evolver
from mtargets.models import Project


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('project_id', nargs=1, type=int)
        parser.add_argument(
            '--num', '-n', default=1, help='How many generations?',
            type=int)
        parser.add_argument(
            '--force', '-f', default=False, action='store_true')

    def handle(self, *args, **options):
        p = Project.objects.get(pk=options['project_id'][0])
        force = options['force']
        should_evolve = False
        if p.last_evolved is None:
            should_evolve = True
        elif p.population.filter(died_at__isnull=True).count() < p.pop_min:
            should_evolve = True
        else:
            item = p.population.order_by('-last_voted').first()
            if item is None:
                should_evolve = True
            elif item.last_voted is None:
                should_evolve = True
            else:
                should_evolve = item.last_voted > p.last_evolved
        if force:
            should_evolve = True
        if not should_evolve:
            print('No evolution required')
            return
        for i in range(options['num']):
            print(f'Generation {i+1}/{options["num"]}')
            evolver = Evolver(options['project_id'][0])
            evolver.evolve()
