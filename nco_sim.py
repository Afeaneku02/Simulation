# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:56:31 2024

@author: Winfred
"""
import numpy as np 
import random as r
import math
import numpy.linalg
import matplotlib.pyplot as plt


def create_adjacency_matrix(num_nodes):
    """ Create an adjacency matrix for a network with a specified number of nodes. 
    Initially, no nodes are connected; the matrix is filled with zeros. 
    Parameters: - num_nodes: The number of nodes in the network. 
    Returns: - A NumPy array representing the adjacency matrix of the network. """
    
    return np.zeros((num_nodes, num_nodes), dtype=int) # Example: Create an adjacency matrix for a network with 5 nodes adjacency_matrix = create_adjacency_matrix(5)

def arbitrary_graph(matrix):
    """
    Create an arbitrary graph which is represented by a fully connected matreix.
    
    Parameters:
    - matrix: The adjacency matrix of the network.
    
    Returns:
    - True if the network is fully connected, False otherwise.
    """
    # how many nodes should I connect 1st???/
     #n-1 nodes 
     #series of n-1 add edges until we reach desitred topology
    
  #  row = r.randint(0, len(matrix)-1) # get a row
    """  here you need to change the number  """
    IsThere_path_to_all_nodes  =np.linalg.matrix_power(matrix,len(matrix))
    while np.count_nonzero(IsThere_path_to_all_nodes)<len(matrix)**2:
   # for i in range(0, len(matrix)):
        row= r.randint(0, len(matrix)-1)
        col = r.randint(0, len(matrix)-1)
    
        while  col == row:
             col = r.randint(0, len(matrix)-1)
        matrix[row, col] = 1 # create a connection/link
        matrix[col, row] = 1
        IsThere_path_to_all_nodes  =np.linalg.matrix_power(matrix,len(matrix))    
    
    return matrix 

def execution(matrix):
    """
    Execute the network simulation until a fully connected topology is achieved.
    
    Parameters:
    - matrix: The adjacency matrix of the network.
    
    Returns:
    - num_rounds: The number of rounds taken to achieve full connectivity.
    - visited_nodes: The connectivity progress over time.
    """
    visited_nodes = [] #subject to removal 3/26
    num_rounds = 0
    count=0
    num_nodes = len(matrix)
    #for i in range(0,3):
     #   check_topology(matrix)
    node_connections_over_time = {i: [] for i in range(len(matrix[0]))} #5/3 plt
    visited_nodes.append(np.sum(matrix) *.1)

    while check_topology(matrix, num_rounds,count) != True:

        sender_node = r.randint(0, len(matrix) -1) 
           
        # By squaring the adjacency matrix, we can determine the nodes that are two steps away from each other.
        # Numpy's matrix multiplication function can be used for this purpose. This is effective for finding all nodes
        # that are a 'distance 2 neighbor'.
        
        squared_matrix = np.matmul(matrix,matrix)
        neighbors_2_steps_away = np.where((squared_matrix[sender_node] > 0) & (matrix[sender_node] == 0))[0]
     
        # If a message is successfully delivered, we can enhance the connectivity of the graph. We achieve this by
        # identifying nodes that are currently at distance 2 and updating them to be at distance 1. Essentially, within
        # a given range, if there is an indirect path (a path of length 2) from one node to another, we can create a
        # direct link (making it a path of length 1), thus connecting the current node more closely to those nodes.
        num_rounds+= 1
        for node in neighbors_2_steps_away:
            if node == sender_node:  # if the nighber that is 2 steps away is the sender node then skip this itteration
                continue
            else:
                matrix[node][sender_node] = 1
                matrix[sender_node][node] = 1
               
    print("end")
    print("it took ",num_rounds, "   rounds")
    return num_rounds, visited_nodes

def check_topology(matrix,num_rounds,count):

    clique_topology = np.ones(( len(matrix), len(matrix)), dtype=int)
    for node in range (0, len(matrix)):
        clique_topology[node][node]= 0
    
  
    return np.array_equal(matrix, clique_topology)
    
def main():
        
    phase1 = create_adjacency_matrix(200)
    phase2 = arbitrary_graph(phase1)
    phase3,node_connections_over_time = execution(phase2)
    plt.figure()
    #for node, connections in node_connections_over_time.items():
    plt.plot(node_connections_over_time)
    
    plt.title('Node Connectivity Over Time')
    plt.xlabel('Number of nodes')
    plt.ylabel('Convergence Time (rounds)')
    
    plt.show()

if __name__ == "__main__":
    main()
    