# -*- coding: utf-8 -*-


import numpy as np
import math

class nas_graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.node_weights = [1, 4, 8, 16] # for node types: 0, 1, 2, 3, respectively.
        self.edge_weight = 10 # for edge types: forward edge / skip connection
        self.G_matrix = np.zeros([num_nodes, num_nodes])
        self.C_matrix = np.zeros([num_nodes, num_nodes])
        
    def get_feature(self, filename): 
        fp = open(filename)
        lines = fp.readlines()
        fp.close()
        
        i = 0
        for line in lines:
            
            if(len(line) == 0):
                continue
                
            cuts = line.split()
            
            node_id = int(cuts[0])
            self.C_matrix[i, i] = 1 / math.sqrt(self.node_weights[node_id])
            for j in range(1, len(cuts)):
                flag = int(cuts[j])
                if(flag):
                    self.G_matrix[i,j-1] += -self.edge_weight
                    self.G_matrix[j-1,i] += -self.edge_weight
                    self.G_matrix[i,i] += self.edge_weight
                    self.G_matrix[j-1,j-1] += self.edge_weight
            
            i = i + 1
        
        G_matrix = self.G_matrix
        print(G_matrix)
        t_matrix = np.dot(self.C_matrix, self.G_matrix)
        A_matrix = np.dot(t_matrix, self.C_matrix)
        print(A_matrix)
        # calculate eigenvalue & eigenvectors
        eigvals, eigvecs = np.linalg.eig(A_matrix)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        
        temp = eigvecs[:, self.num_nodes-3:self.num_nodes-1]
        x = abs(np.reshape(temp.transpose(), [2*self.num_nodes, -1]))
        
        return x
    
if __name__ == '__main__':
    g = nas_graph(6)
    x1 = g.get_feature('nas1.txt')
    g = nas_graph(6)
    x2 = g.get_feature('nas2.txt')
    g = nas_graph(6)
    x3 = g.get_feature('nas3.txt')
    print(x1,x2,x3)
    print(np.linalg.norm(x1-x2, ord=2))
    print(np.linalg.norm(x1-x3, ord=2))

        