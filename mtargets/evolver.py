import random
from itertools import cycle
import numpy as np

from django.utils import timezone

from mtargets.models import Project, Genotype


class Evolver:

    def __init__(self, project_id):
        self.project = Project.objects.get(pk=project_id)

    def evolve(self):
        new_pop = []
        # Crossover & mutate
        new_pop.extend(self.crossover())
        # Add newbies to population
        new_pop.append(self.generate_genotype())
        # Re-calcuation fitness
        self.project.recalc_fitness()
        # Reduce population to maximum
        self.project.reduce_population(len(new_pop))
        # Save new population
        for gt in new_pop:
            gt.born_gen = self.project.generations + 1
            gt.save()  # make sure we have a PK
            gt.render()
            gt.save() # save again
        self.project.generations += 1
        self.project.last_evolved = timezone.now()
        self.project.save()

    def generate_genotype(self):
        sequence = list(
            random.randint(0, 2**self.project.gene_bits - 1)
            for x in range(self.project.initial_sequence_length))
        return Genotype(project=self.project, sequence=sequence)

    def crossover(self):
        offsprings = []
        parents = list(self.project.get_fittest())
        while len(offsprings) < self.project.crossover_max:
            mom, dad = self._co_select_parents(parents)
            if not (mom or dad):
                break
            offsprings.append(self._co_run(mom, dad))
        return offsprings

    def _co_select_parents(self, candidates):
        if len(candidates) < 2:
            return None, None
        def _sel_one():
            probs = np.array([c.fitness for c in candidates])
            probs = probs / probs.sum()
            return np.random.choice(candidates, p=probs)
        mom = _sel_one()
        candidates.remove(mom)
        dad = _sel_one()
        candidates.remove(dad)
        return mom, dad

    def _co_run(self, mom, dad):
        assert mom is not None and dad is not None
        target_length = np.random.randint(
            max(self.project.seq_min_length,
                min(len(mom.sequence), len(dad.sequence)) - 10),
            min(self.project.seq_max_length,
                max(len(mom.sequence), len(dad.sequence)) + 10))
        
        seq = []
        mseq = cycle(mom.sequence)
        dseq = cycle(dad.sequence)
        f = [mom.fitness, dad.fitness]
        probs = np.array(f) / np.sum(f) 
        while len(seq) < target_length:
            g_mom = next(mseq)
            g_dad = next(dseq)
            seq.append(int(np.random.choice([g_mom, g_dad], p=probs)))
        seq = self._mutate(seq)
        return Genotype(
            project=self.project, sequence=seq,
            parent1=mom, parent2=dad)

    def _mutate(self, sequence):
        # select n% of the sequence as indices
        all_indices = np.arange(len(sequence))
        n_mut = round(len(sequence) * self.project.mutation_rate)
        for idx in np.random.choice(all_indices, size=n_mut):
            b_gene = list(format(
                sequence[idx], f'0{self.project.gene_bits}b'))
            affected_base = np.random.randint(len(b_gene))
            if b_gene[affected_base] == '1':
                b_gene[affected_base] = '0'
            else:
                b_gene[affected_base] = '1'
            sequence[idx] = int(''.join(b_gene), 2)
        return sequence
            
