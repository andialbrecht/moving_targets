from django.contrib import admin
from django.db.models import Count, F

from mtargets import models


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'generations', 'last_evolved')

admin.site.register(models.Project, ProjectAdmin)


class GenotypeAdmin(admin.ModelAdmin):
    list_display = ('pk', 'project', 'fitness', 'born_at', 'died_at',
                    'born_gen', 'died_gen', 'sequence_length',
                    'num_children',
                    'born_universe', 'died_universe')
    list_filter = ('project',)

    def get_queryset(self, request):
        query = super().get_queryset(request)
        query = query.annotate(c1_count=Count('child1'))
        query = query.annotate(c2_count=Count('child2'))
        query = query.annotate(num_children=F('c1_count') + F('c2_count'))
        return query

    def num_children(self, obj):
        return obj.num_children
    num_children.admin_order_field = 'num_children'

    def born_universe(self, obj):
        # Just for fun, given them some "real" birthdays
        # "real" in the terms of the universe they're living in.
        # Maybe names also
        if not obj.born_at:
            return None
        return obj.born_at.timestamp() // 600 - 2685300

    def died_universe(self, obj):
        if not obj.died_at:
            return None
        return obj.died_at.timestamp() // 600 - 2685300

admin.site.register(models.Genotype, GenotypeAdmin)
