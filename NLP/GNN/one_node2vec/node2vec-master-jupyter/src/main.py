'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")
	# 输入文件地址
	parser.add_argument('--input', nargs='?', default='../graph/karate.edgelist',
	                    help='Input graph path')
	# 输出的结点向量地址
	parser.add_argument('--output', nargs='?', default='../emb/karate_v2.emb',
	                    help='Embeddings path')
	# 向量的维度
	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')
	# 每一次采样游走的步数
	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')
	# 以每个点为起始点游走次数
	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')
	# word2vec训练窗口大小
	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')
	# word2vec训练的轮数
	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')
	# word2vec启动并行工作结点的数量（cpu计算数量）
	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')
	# bias random walk 中参数p
	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')
	# bias random walk 中参数q
	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')
	# 默认设置边权重都是1
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	# 默认这是都是无向图
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()


def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	# ret_walks = []
	# for walk in walks:
	# 	ret_walks.append([str(w) for w in walk])
	# walks = ret_walks

	walks = [list(map(str, walk)) for walk in walks]
	print(len(walks))
	print(len(walks[0]))
	ret_list = []
	for walk in walks:
		ret_list.append(walk[0])
	ret_list.sort()
	print(ret_list)

	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return


def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)


if __name__ == "__main__":
	args = parse_args()
	main(args)
