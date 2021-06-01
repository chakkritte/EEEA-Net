from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7',
    'conv_1x1_3x3',
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

SNAS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                        ('skip_connect', 0), ('dil_conv_3x3', 1),
                        ('skip_connect', 1), ('skip_connect', 0), 
                        ('skip_connect',0),  ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                 ('max_pool_3x3', 1), ('skip_connect', 2),
                 ('skip_connect', 2), ('max_pool_3x3', 1),
                 ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
PC_DARTS = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
CDARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
CARS_I = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
CARS_H = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

DARTS = DARTS_V2

#NUEABest
EEEA = Genotype(normal=[('sep_conv_3x3', 0), ('inv_res_5x5', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('inv_res_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('inv_res_5x5', 0), ('dil_conv_5x5', 1)], reduce_concat=range(3, 6))

NU_EA = Genotype(normal=[('sep_conv_3x3', 1), ('conv_1x1_3x3', 0), ('sep_conv_5x5', 0), ('conv_1x1_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))  

#2.883214MB
Normal_Best_All = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('inv_res_5x5', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 1)], normal_concat=range(2, 6), reduce=[('inv_res_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 2), ('skip_connect', 0)], reduce_concat=range(3, 6))

#2.119294MB
EE_Best_All = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)], normal_concat=range(3, 6), reduce=[('inv_res_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_5x5', 2), ('inv_res_5x5', 2), ('dil_conv_3x3', 2), ('skip_connect', 1)], reduce_concat=range(3, 6))

#2.343286MB
Ultra_Best_All = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('inv_res_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(3, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('inv_res_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

#2.162494MB
Ultra_Best_All_50 = Genotype(normal=[('inv_res_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=range(3, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0)], reduce_concat=range(3, 6))

G51405ac310f88375ed7e9b9fa7bfdce0 = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('inv_res_5x5', 3), ('inv_res_3x3', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 3)], normal_concat=[5, 6], reduce=[('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('inv_res_5x5', 1), ('dil_conv_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 3), ('skip_connect', 3), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)], reduce_concat=[2, 4, 5, 6])

Gc25f061acd62d517ccacfd6685d16b99 = Genotype(normal=[('max_pool_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('inv_res_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 4), ('skip_connect', 3)], normal_concat=[5, 6], reduce=[('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 0)], reduce_concat=[4, 5, 6])

Gb728e3edb4140725a222c6e2df7effeb = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('sep_conv_3x3', 4)], normal_concat=[5, 6], reduce=[('skip_connect', 0), ('inv_res_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('inv_res_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 3)], reduce_concat=[4, 5, 6])

EEEA_L = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 2), ('inv_res_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[3, 5, 6], reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 3), ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

G1a8eaca4c95297f5c5a2ae2d7be028e6 = Genotype(normal=[('avg_pool_3x3', 0), ('mbconv_k3_t1', 0), ('avg_pool_3x3', 1), ('mbconv_k3_t1', 1), ('mbconv_k3_t1', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 1), ('mbconv_k5_t1', 3)], normal_concat=[4, 5], reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('mbconv_k3_t1', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=[3, 4, 5])

#version2
#normal cifar10 1 epoch
G1cf2d752194ac646828fb5b523ccbfbb = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('mbconv_k5_t1', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=[4, 5, 6], reduce=[('inv_res_5x5_sh', 0), ('dil_conv_3x3', 0), ('blur_pool_3x3', 1), ('avg_pool_3x3', 0), ('std_gn_3x3', 0), ('inv_res_3x3', 1), ('mbconv_k5_t1', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('inv_res_3x3_sh', 4)], reduce_concat=[2, 3, 5, 6])
#ee 5.0 cifar10 1 epoch
G15d4950b9a555af52bf9d146e0c3bbd5 = Genotype(normal=[('blur_pool_3x3', 0), ('std_gn_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('mbconv_k3_t1', 1), ('mbconv_k3_t1', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 4)], normal_concat=[3, 5, 6], reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('inv_res_5x5', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('std_gn_3x3', 3), ('mbconv_k5_t1', 1), ('conv_7x1_1x7', 2)], reduce_concat=[4, 5, 6])

#normal cifar100 1 epoch
G87d1f3a6b7536cb1716c08512db35207 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('inv_res_5x5', 2)], normal_concat=[4, 5, 6], reduce=[('mbconv_k5_t1', 0), ('std_gn_5x5', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_7x7', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('conv_7x1_1x7', 0), ('dil_conv_3x3', 5)], reduce_concat=[3, 4, 6])
#ee 5.0 cifar100 1 epoch
G7f15ccadf94f34b9208c51079f5731f8 = Genotype(normal=[('blur_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('blur_pool_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 2), ('blur_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 4)], normal_concat=[3, 5, 6], reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('std_gn_3x3', 0), ('inv_res_5x5', 0), ('skip_connect', 1), ('inv_res_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('conv_7x1_1x7', 4), ('avg_pool_3x3', 2)], reduce_concat=[5, 6])

# p-eeea normal cifar10 1 epoch
G5c093e5f8cb17109b27720acfe386167 = Genotype(normal=[('std_gn_3x3', 0), ('dil_conv_3x3', 0), ('mbconv_k5_t1', 1), ('skip_connect', 1), ('mbconv_k3_t1', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=[5, 6], reduce=[('inv_res_3x3_sh', 0), ('avg_pool_3x3', 0), ('std_gn_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('std_gn_5x5', 2), ('conv_7x1_1x7', 2), ('max_pool_3x3', 3), ('std_gn_3x3', 0), ('std_gn_3x3', 0)], reduce_concat=[4, 5, 6])
# p-eeea ee 5.0 cifar10 1 epoch

G6dd9293230e5836c71e69907c9256627 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 0), ('none', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('none', 3)], reduce_concat=[4, 5])

Ga518053bb620759b53b8d78029dad591 = Genotype(normal=[('max_pool_3x3', 0), ('none', 0), ('skip_connect', 0), ('none', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 3)], normal_concat=[2, 4, 5], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 3), ('none', 3)], reduce_concat=[4, 5])

G769cf3fe714587d6482ccfd3755f3886 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 1), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
output = G769cf3fe714587d6482ccfd3755f3886
