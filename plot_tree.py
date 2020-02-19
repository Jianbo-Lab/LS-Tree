from nonlinearity import find_roots, find_depth
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
def find_pos(adj_list, words, width = 3.0, height = 1.0):
	depths = find_depth([adj_list])
	depth_to_nodes = {}
	pos = {} 
	for node in depths:
		if depths[node] in depth_to_nodes:
			depth_to_nodes[depths[node]].append(node)
		else:
			depth_to_nodes[depths[node]] = [node]

	depths = depth_to_nodes.keys()
	num_layers = max(depths)
	for depth in np.arange(1,max(depths)+1):
		nodes_at_depth = sorted(depth_to_nodes[depth], key = lambda node: node[0])
		h_at_depth = depth / (float(num_layers)+1) * height

		num_nodes = len(nodes_at_depth)
		for j, node in enumerate(nodes_at_depth):
			# print(node)
			if depth == 1:
				w_at_node = (j+1) / (float(num_nodes)+1) * width
			else:
				w_at_node = np.mean([pos[child][0] for child in adj_list[node]])

			pos[node] = (w_at_node, h_at_depth)

	num_words = len(words)
	for j, word in enumerate(words):
		pos[word] = ((j+1) / (float(num_words)+1) * width, 1 / (float(num_layers)+1) * height - 0.1)

	return pos

def create_graph(adj_lists, words, node_to_d, title, labels = {}):
	for tree_id, children_list in enumerate(adj_lists): 
		start = min([node[0] for node in children_list])
		end = max([node[1] for node in children_list])
		words_in_tree = words[start:end]
		G=nx.Graph()
		G.add_nodes_from(children_list.keys())
		G.add_nodes_from(words_in_tree)
		for start_node in children_list:
			for end_node in children_list[start_node]:
				G.add_edge(start_node, end_node)
		# pos =graphviz_layout(G, prog='dot')
		# pos = hierarchy_pos(G, root) 
		pos = find_pos(children_list, words_in_tree)   
		labels_in_tree = {node: labels[node] for node in children_list if node in labels}
		for word in words_in_tree:
			labels_in_tree[word] = word

		colors = []
		for node in list(G.nodes()):
			if node in node_to_d: 
				# color = np.minimum(np.maximum(-1, node_to_d[node][1]), 1)
				color = node_to_d[node][1]
				colors.append(color)
			else:
				colors.append(0)

		colors = np.array(colors)
		colors = colors / (np.max(abs(colors))+1e-8)
		plt.figure(figsize = (15,5))
		nx.draw_networkx(G, pos, with_labels = False, node_color = colors, cmap = plt.get_cmap('bwr'), 
			vmin = -1,
			vmax = 1)
		nx.draw_networkx_labels(G, pos, labels_in_tree,font_size=10)
		locs = [pos[node] for node in children_list]
		plt.scatter([loc[0] for loc in locs], [loc[1] for loc in locs], 
			s = [300 for loc in locs],
			linewidths = 3, edgecolor = 'black')
		plt.axis('off')
		if not os.path.exists('figs'):
			os.mkdir('figs')

		if not os.path.exists('figs/demo'):
			os.mkdir('figs/demo')
		plt.savefig("figs/demo/{}-{}.pdf".format(title, tree_id), bbox_inches='tight')
		plt.close()








