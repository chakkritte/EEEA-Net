from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

#MOEA
MOEA = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('mbconv_k3_t1', 1), ('mbconv_k3_t1', 0), ('skip_connect', 2), ('inv_res_3x3', 3), ('mbconv_k5_t1', 3)], normal_concat=[4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 2)], reduce_concat=[3, 4, 5])

# MOEA SOTA-PI
MOEA_SOTA_PI = Genotype(normal=[('avg_pool_3x3', 0), ('inv_res_5x5', 0), ('inv_res_3x3', 0), ('mbconv_k3_t1', 1), ('inv_res_3x3_sh', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=[3, 4, 5], reduce=[('mbconv_k5_t1', 0), ('max_pool_3x3', 0), ('std_gn_3x3', 0), ('inv_res_3x3_sh', 1), ('mbconv_k3_t1', 2), ('skip_connect', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1)], reduce_concat=[3, 4, 5])

EEEAL = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('inv_res_5x5', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 2), ('inv_res_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[3, 5, 6], reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 3), ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])
