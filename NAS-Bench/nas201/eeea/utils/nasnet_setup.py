import numpy as np
from collections import namedtuple
from eeea.network.genotypes import Structure

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# def setup_NAS(n_blocks, n_ops):
#     n_var = int(4 * n_blocks * 2)
#     ub = np.ones(n_var)
#     h = 1
#     for b in range(0, n_var//2, 4):
#         ub[b] = n_ops
#         ub[b + 1] = h
#         ub[b + 2] = n_ops
#         ub[b + 3] = h
#         h += 1
#     ub[n_var//2:] = ub[:n_var//2]
#     return ub

# for NAS-Bench-201
def setup_NAS(n_ops):
    result = np.empty(6)
    result.fill(n_ops)
    return result

# def create_param_choices(primitives, nas_setup):
#     nn_param_choices = {}
#     for i in range(len(nas_setup)):
#         if i % 2 == 0:
#             nn_param_choices[str(i)] = primitives
#         else:
#             end_index = int(nas_setup[i])
#             nn_param_choices[str(i)] = np.arange(end_index).tolist()
#     return nn_param_choices

# for NAS-Bench-201
def create_param_choices(primitives, nas_setup):
    nn_param_choices = {}
    for i in range(len(nas_setup)):
        nn_param_choices[str(i)] = primitives
    return nn_param_choices


def decode(genotype):
    output = []
    for i in range(3):
        if i == 0:
            x_list = [(genotype['0'], 0)]
            output.append( tuple(x_list) )
        elif i == 1:
            x_list = [(genotype['1'], 0), (genotype['2'], 1)]
            output.append( tuple(x_list) )
        elif i == 2:
            x_list = [(genotype['3'], 0), (genotype['4'], 1), (genotype['5'], 2)]
            output.append( tuple(x_list) )
    return Structure(output)


def decode_cell(chromosome):
    normal_cell = []
    reduce_cell = []
    size = int(len(chromosome)/2)
    count = 0
    for key,val in chromosome.items():
        if count < size:
            normal_cell.append(val)
        else:
            reduce_cell.append(val)    
        count += 1

    normal, normal_concat = [], list(range(2, int(len(normal_cell)/4)+2))
    reduce, reduce_concat = [], list(range(2, int(len(reduce_cell)/4)+2))

    for i in range(len(normal_cell)):
      if i % 2 == 1:
        normal.append((normal_cell[i-1], normal_cell[i]))
        #if isinstance(normal_cell[i], int) and normal_cell[i] in normal_concat:
        #  normal_concat.remove(normal_cell[i])

    for i in range(len(reduce_cell)):
      if i % 2 == 1:
        reduce.append((reduce_cell[i-1], reduce_cell[i]))
        #if isinstance(reduce_cell[i], int) and reduce_cell[i] in reduce_concat:
        #  reduce_concat.remove(reduce_cell[i])

    return Genotype(normal=normal,
                            normal_concat= normal_concat,
                            reduce=reduce,
                            reduce_concat= reduce_concat)

def decode_genotype(genotype):
    network = {}
    index = 0
    for key in genotype.normal:
        network[str(index)]=key[0]
        network[str(index+1)]=key[1]
        index += 2
    for key in genotype.reduce:
        network[str(index)]=key[0]
        network[str(index+1)]=key[1]
        index += 2
    return network