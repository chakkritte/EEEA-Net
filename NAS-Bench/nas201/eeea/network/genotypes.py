from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

from copy import deepcopy


def get_combination(space, num):
  combs = []
  for i in range(num):
    if i == 0:
      for func in space:
        combs.append( [(func, i)] )
    else:
      new_combs = []
      for string in combs:
        for func in space:
          xstring = string + [(func, i)]
          new_combs.append( xstring )
      combs = new_combs
  return combs
  

class Structure:

  def __init__(self, genotype):
    assert isinstance(genotype, list) or isinstance(genotype, tuple), 'invalid class of genotype : {:}'.format(type(genotype))
    self.node_num = len(genotype) + 1
    self.nodes    = []
    self.node_N   = []
    for idx, node_info in enumerate(genotype):
      assert isinstance(node_info, list) or isinstance(node_info, tuple), 'invalid class of node_info : {:}'.format(type(node_info))
      assert len(node_info) >= 1, 'invalid length : {:}'.format(len(node_info))
      for node_in in node_info:
        assert isinstance(node_in, list) or isinstance(node_in, tuple), 'invalid class of in-node : {:}'.format(type(node_in))
        assert len(node_in) == 2 and node_in[1] <= idx, 'invalid in-node : {:}'.format(node_in)
      self.node_N.append( len(node_info) )
      self.nodes.append( tuple(deepcopy(node_info)) )

  def tolist(self, remove_str):
    # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
    # note that we re-order the input node in this function
    # return the-genotype-list and success [if unsuccess, it is not a connectivity]
    genotypes = []
    for node_info in self.nodes:
      node_info = list( node_info )
      node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
      node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
      if len(node_info) == 0: return None, False
      genotypes.append( node_info )
    return genotypes, True

  def node(self, index):
    assert index > 0 and index <= len(self), 'invalid index={:} < {:}'.format(index, len(self))
    return self.nodes[index]

  def tostr(self):
    strings = []
    for node_info in self.nodes:
      string = '|'.join([x[0]+'~{:}'.format(x[1]) for x in node_info])
      string = '|{:}|'.format(string)
      strings.append( string )
    return '+'.join(strings)

  def check_valid(self):
    nodes = {0: True}
    for i, node_info in enumerate(self.nodes):
      sums = []
      for op, xin in node_info:
        if op == 'none' or nodes[xin] is False: x = False
        else: x = True
        sums.append( x )
      nodes[i+1] = sum(sums) > 0
    return nodes[len(self.nodes)]

  def to_unique_str(self, consider_zero=False):
    # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
    # two operations are special, i.e., none and skip_connect
    nodes = {0: '0'}
    for i_node, node_info in enumerate(self.nodes):
      cur_node = []
      for op, xin in node_info:
        if consider_zero is None:
          x = '('+nodes[xin]+')' + '@{:}'.format(op)
        elif consider_zero:
          if op == 'none' or nodes[xin] == '#': x = '#' # zero
          elif op == 'skip_connect': x = nodes[xin]
          else: x = '('+nodes[xin]+')' + '@{:}'.format(op)
        else:
          if op == 'skip_connect': x = nodes[xin]
          else: x = '('+nodes[xin]+')' + '@{:}'.format(op)
        cur_node.append(x)
      nodes[i_node+1] = '+'.join( sorted(cur_node) )
    return nodes[ len(self.nodes) ]

  def check_valid_op(self, op_names):
    for node_info in self.nodes:
      for inode_edge in node_info:
        #assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
        if inode_edge[0] not in op_names: return False
    return True

  def __repr__(self):
    return ('{name}({node_num} nodes with {node_info})'.format(name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__))

  def __len__(self):
    return len(self.nodes) + 1

  def __getitem__(self, index):
    return self.nodes[index]

  @staticmethod
  def str2structure(xstr):
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
      genotypes.append( input_infos )
    return Structure( genotypes )

  @staticmethod
  def str2fullstructure(xstr, default_name='none'):
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = list( (op, int(IDX)) for (op, IDX) in inputs)
      all_in_nodes= list(x[1] for x in input_infos)
      for j in range(i):
        if j not in all_in_nodes: input_infos.append((default_name, j))
      node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
      genotypes.append( tuple(node_info) )
    return Structure( genotypes )

  @staticmethod
  def gen_all(search_space, num, return_ori):
    assert isinstance(search_space, list) or isinstance(search_space, tuple), 'invalid class of search-space : {:}'.format(type(search_space))
    assert num >= 2, 'There should be at least two nodes in a neural cell instead of {:}'.format(num)
    all_archs = get_combination(search_space, 1)
    for i, arch in enumerate(all_archs):
      all_archs[i] = [ tuple(arch) ]
  
    for inode in range(2, num):
      cur_nodes = get_combination(search_space, inode)
      new_all_archs = []
      for previous_arch in all_archs:
        for cur_node in cur_nodes:
          new_all_archs.append( previous_arch + [tuple(cur_node)] )
      all_archs = new_all_archs
    if return_ori:
      return all_archs
    else:
      return [Structure(x) for x in all_archs]

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
