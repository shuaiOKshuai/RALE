import networkx as nx

# adjlist_dict = dict()
# with open('./dataset/reddit/graph.edge', 'r') as f:
#     lines = f.readlines()
#     for l in lines:
#         temp = list(l.strip('\n').split('\t'))
#         if temp[0] not in adjlist_dict.keys():
#             adjlist_dict[temp[0]] = set()
#         adjlist_dict[temp[0]].add(temp[1])
# # with open('./dataset/reddit/graph.adjlist', 'w') as fw:
# #     for n in adjlist_dict.keys():
# #         fw.write(str(n))
# #         for adj in adjlist_dict[n]:
# #             fw.write(' ' + str(adj))
# #         fw.write('\n')
#
# g = nx.read_adjlist('./dataset/reddit/graph.adjlist')
# print(nx.is_connected(g))
# d_comp = list(nx.connected_components(g))
# d_comp.sort(reverse=True)
# print(d_comp)
# print(len(d_comp[0]))
# d_nodes = set()
# for i in range(1, len(d_comp)):
#     for n in d_comp[i]:
#         d_nodes.add(str(n))
# for n in d_nodes:
#     if n in adjlist_dict.keys():
#         del adjlist_dict[n]
# for n in adjlist_dict.keys():
#     for adj in adjlist_dict[n]:
#         if adj in d_nodes:
#             adjlist_dict[n].remove(adj)
# id_map = dict()
# i = 0
# for k in adjlist_dict.keys():
#     id_map[k] = i
#     i += 1
# with open('./dataset/reddit/id_map.txt', 'w') as fw:
#     for k in id_map.keys():
#         fw.write(str(k) + ' ' + str(id_map[k]) + '\n')

id_map = dict()
with open('./dataset/reddit/id_map.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        temp = list(l.strip('\n').split(' '))
        id_map[temp[0]] = temp[1]
with open('./dataset/reddit/graph_new.edge', 'w') as fw:
    with open('./dataset/reddit/graph.edge', 'r') as f:
        lines = f.readlines()
        for l in lines:
            temp = list(l.strip('\n').split('\t'))
            if temp[0] not in id_map.keys() or temp[1] not in id_map.keys():
                continue
            fw.write(id_map[temp[0]] + '\t' + id_map[temp[1]] + '\n')
with open('./dataset/reddit/graph_new.node', 'w') as fw:
    with open('./dataset/reddit/graph.node', 'r') as f:
        lines = f.readlines()
        for l in lines:
            temp = list(l.strip('\n').split('\t'))
            if len(temp) == 2:
                fw.write(str(len(id_map)) + '\t' + temp[1] + '\n')
            if len(temp) > 2:
                if temp[0] in id_map.keys():
                    fw.write(id_map[temp[0]])
                    for ind in range(1, len(temp)):
                        fw.write('\t' + temp[ind])
                    fw.write('\n')
