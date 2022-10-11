import numpy as np

def initialize_population(params):
    pop_size = params['pop_size']
    max_length = params['max_length']
    max_stride2 = params['max_stride2']
    image_channel = params['image_channel']
    max_output_channel = params['max_output_channel']
    population = []
    for _ in range(pop_size):
        num_net = int(max_length)

        part2 = []
        p_valid = np.random.random()
        for i in range(num_net):
            p_ = np.random.random()
            if p_ <= p_valid:
                num_feature_maps = np.random.randint(0, max_output_channel)
            else:
                p1 = np.random.random()
                if p1 <= 0.5:
                    num_feature_maps = np.random.randint(-30, 0)
                else:
                    num_feature_maps = np.random.randint(max_output_channel, max_output_channel+30)
            part2.append(num_feature_maps)


        num_stride2 = np.random.randint(0, max_stride2 + 1)
        num_stride1 = num_net - num_stride2
        # find the position where the pooling layer can be connected
        availabel_positions = list(idx for idx in range(0, num_net) if 0<=part2[idx]<max_output_channel) #only consider those valid layers as possible strided layers
        if len(availabel_positions) == 0:
            part2[0] = np.random.randint(0, max_output_channel)
            availabel_positions = [0]
        np.random.shuffle(availabel_positions)
        np.random.shuffle(availabel_positions)
        while len(availabel_positions) < num_stride2:
            supp_list = [idx for idx in range(0, num_net) if idx not in availabel_positions]
            availabel_positions.append(np.random.choice(supp_list, 1)[0])
        select_positions = np.sort(availabel_positions[0:num_stride2])  # the positions of pooling layers in the net
        part1 = []
        for i in range(num_net):
            if i in select_positions:
                code_stride2 = np.random.randint(3, 6)
                part1.append(code_stride2)
            else:
                code_stride1 = np.random.randint(0, 3)
                part1.append(code_stride1)

        population.append([part1, part2])
    return population

def test_population():
    params = {}
    params['pop_size'] = 20
    params['max_length'] = 30
    params['max_stride2'] = 4
    params['image_channel'] = 1
    params['max_output_channel'] = 256
    pop = initialize_population(params)
    print(pop)

if __name__ == '__main__':
    test_population()
