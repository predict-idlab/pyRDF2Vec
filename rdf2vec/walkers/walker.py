

class Walker():
    def __init__(self, depth, walks_per_graph):
        self.depth = depth
        self.walks_per_graph = walks_per_graph

    def print_walks(self, graph, instances, file_name):
        walks = self.extract(graph, instances)
        walk_strs = []
        for walk_nr, walk in enumerate(walks):
            s = ''
            for i in range(len(walk)):
                if i % 2:
                    s += '{} '.format(walk[i])
                else:
                    s += '{} '.format(walk[i])
                
                if i < len(walk) - 1:
                    s += '--> '

            walk_strs.append(s)

        with open(file_name, "w+") as myfile:
            for s in walk_strs:
                myfile.write(s)
                myfile.write('\n\n')

    def extract(self, graph, instances):
        raise NotImplementedError('This must be implemented!')
