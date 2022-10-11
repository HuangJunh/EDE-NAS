import copy

import numpy as np

class FDE(object):
    def __init__(self, individual, idx, population, acc_set, params, _log):

        self.individual = individual
        self.idx = idx
        self.population = population
        self.acc_set = acc_set
        self.params = params
        self.diff_rate = params['diff_rate']
        self.crossover_rate = params['crossover_rate']
        self.max_stride2 = params['max_stride2']
        self.log = _log

    def obtain_better_inds_idxs(self):
        acc_cur = self.acc_set[self.idx]
        idxs = []
        for idx in range(len(self.acc_set)):
            if not idx == self.idx:
                if self.acc_set[idx] >= acc_cur:
                    idxs.append(idx)

    def mutate(self):
        better_inds_idxs = self.obtain_better_inds_idxs()
        if better_inds_idxs:
            id0 = np.random.choice(better_inds_idxs, 1)[0]
            x0 = np.asarray(self.population[id0])
        else:
            x0 = np.asarray(self.individual)
        pop_size = len(self.population)
        idxs = np.random.choice(list(range(pop_size)), 2, replace=False)
        while self.idx in idxs:
            idxs = np.random.choice(list(range(pop_size)), 2, replace=False)
        x1 = np.asarray(self.population[idxs[0]])
        x2 = np.asarray(self.population[idxs[1]])

        #part1 mutation
        indi_mut_1 = []
        diff_v = [x1[0][i] if not x1[0][i]==x2[0][i] else 0 for i in range(len(x1[0]))]
        for i in range(len(x0[0])):
            p_ = np.random.random()
            if p_ <= self.diff_rate:
                if diff_v[i] == 0:
                    rand_opt = np.random.randint(0, 6)
                    indi_mut_1.append(rand_opt)
                else:
                    indi_mut_1.append(diff_v[i])
            else:
                indi_mut_1.append(x0[0][i])

        #part2 mutation
        indi_mut_2 = np.asarray(self.individual[1]) + self.diff_rate*(x0[1] - np.asarray(self.individual[1])) + self.diff_rate*(x1[1] - x2[1])
        indi_mut_2 = list(map(int, indi_mut_2))

        return [indi_mut_1, indi_mut_2]

    def crossover(self, indi_mut):
        offspring_part1 = []
        offspring_part2 = []
        [indi_mut_1, indi_mut_2] = indi_mut
        j = np.random.choice(len(self.individual[0]))
        for i in range(len(self.individual[0])):
            p_ = np.random.random()
            if p_ <= self.crossover_rate or i==j:
                offspring_part1.append(indi_mut_1[i])
                offspring_part2.append(indi_mut_2[i])
            else:
                offspring_part1.append(self.individual[0][i])
                offspring_part2.append(self.individual[1][i])
        indi_mut_ = self.adjust_Indi([offspring_part1, offspring_part2])
        return indi_mut_

    def get_valid_indi(self, individual):
        valid_part1 = []
        valid_part2 = []
        for i, element in enumerate(individual[1]):
            if 0<=element<self.params['max_output_channel']:
                valid_part1.append(individual[0][i])
                valid_part2.append(element)
        return [valid_part1, valid_part2]

    def adjust_Indi(self, individual):
        valid_indi = self.get_valid_indi(individual)
        if len(valid_indi[0]) == 0:
            individual[0][0] = np.random.randint(0, 6)
            individual[1][0] = np.random.randint(0, self.params['max_output_channel'])
            return individual
        else:
            valid_pos_stride2 = self._count_stride2(valid_indi)
            pos_stride2 = self._count_stride2(individual)
            # stride2_idxs = [i for i in range(len(pos_stride2)) if pos_stride2[i]==1]
            stride2_idxs = [i for i in range(len(pos_stride2)) if pos_stride2[i] == 1 and 0<=individual[1][i]<self.params['max_output_channel']]
            part1 = individual[0]
            part2 = individual[1]
            individual = copy.deepcopy(individual)
            if sum(valid_pos_stride2) > self.max_stride2:
                # num2cut = np.random.randint(sum(pos_stride2)-self.max_stride2, sum(pos_stride2)-self.max_stride2+2)
                num2cut = sum(valid_pos_stride2)-self.max_stride2
                selected_idx = np.random.choice(stride2_idxs, num2cut, replace=False)
                for idx in selected_idx:
                    individual[0][idx] = part1[idx]-3
            for i, element in enumerate(part2):
                if element > self.params['max_output_channel'] + 30:
                    individual[1][i] = self.params['max_output_channel'] + 30
                elif element < -30:
                    individual[1][i] = -30
            return individual

    def _count_stride2(self, individual):
        pos_stride2 = [0]*(len(individual[0]))
        for i, element in enumerate(individual[0]):
            if element >= 3:
                pos_stride2[i] = 1
        return pos_stride2




