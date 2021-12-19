from django.http import JsonResponse, HttpResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.views.generic.detail import DetailView

from mtargets import models


class ProjectView(DetailView):
    model = models.Project

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        query = context['object'].population.all()
        query = query.filter(died_at__isnull=True)
        q_order = self.request.GET.get('o')
        if q_order == 'n':
            query = query.order_by('-born_at')
        else:
            query = query.order_by('-fitness')
        context['population'] = query
        return context


class GenotypeView(DetailView):
    model = models.Genotype


def get_image(request):
    # TODO: Add error handling for missing or invalid URL parameters
    last_id = int(request.GET.get('i'))
    vote = int(request.GET.get('v'))
    g = models.Genotype.objects.get(pk=last_id)
    if vote == 1:
        g.fitness = (g.fitness + 1) / 2
    else:
        g.fitness = g.fitness * 0.5
    g.last_voted = timezone.now()
    g.save()
    project = models.Project.objects.get(pk=1)
    item = project.one_item()
    response = JsonResponse({
        'id': item.pk,
        'url': item.image_url()
    })
    return response


def vote(request, **kwargs):
    g = models.Genotype.objects.get(pk=kwargs['pk'])
    if kwargs['what'] == 1:
        g.fitness = (g.fitness + 1) / 2
    else:
        g.fitness = g.fitness * 0.5
    g.last_voted = timezone.now()
    g.save()
    next_item = g.project.one_item()
    return redirect(f'/g/{next_item.pk}/')


def download_scaled(request, **kwargs):
    g = models.Genotype.objects.get(pk=kwargs['pk'])
    data = g.render_stream(kwargs['w'], kwargs['h'])
    response = HttpResponse(data, content_type='image/jpeg')
    return response
