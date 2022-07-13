
import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt
import json
import copy
import os

def multivariate_gaussian( mu, Sigma,scale):
    mu = np.array([mu[0]/scale[0],mu[1]/scale[1]])
    x, y = np.mgrid[0:1:1/scale[0], 0:1:1/scale[1]]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu, Sigma)
    return rv.pdf(pos)


class TrafficPaths:

    """
    path_distribution - distribution of lengths of paths (spatial, not graph)
    number_cores -  number of core areas in the network
    lattice_shape - (x number of edges, y number of edges) for each side of 2d lattice
    length_unit - lengtht of an edge. Only tried length 1
    unitball_delta - thickness of a concentric layer within which targets are allowed 
    """

    def __init__(self, path_distribution, number_cores, lattice_shape,\
                 length_unit,unitball_delta,**kwargs ):
        self.path_distribution =  path_distribution(**kwargs)
        # self.path_distribution  = (self.path_distribution -min(self.path_distribution ))/(max(self.path_distribution )-self.path_distribution )
        
        self.length_unit = length_unit
        self.unitball_delta = unitball_delta
        self.number_cores = number_cores
        self.lattice_shape = lattice_shape
        self.paths = []
        self.path_lengths_expected = []
        self.path_lengths_executed = []
        self.graph = self.generate_lattice()
        self.sources = []
        
    def generate_lattice(self):
        #generate a lattice graph of shape lattice_shape
        G =nx.grid_2d_graph(self.lattice_shape[0],self.lattice_shape[1])
        
        #if there are "core areas" in the graph, i.e. more central areas 
        # for each we generate a 2D gaussian,  which we can use to assign weights to edges
        if self.number_cores !=0:
            #2D gaussian is stored in path_density
            path_density = np.ones((self.lattice_shape[0],self.lattice_shape[1]))
            x, y = [],[]
            for i in range(self.number_cores):
                #randomly sample coordinate for the core
                # xcoord,ycoord = (np.random.randint(0,self.lattice_shape[0]),np.random.randint(0,self.lattice_shape[1]))
                xcoord,ycoord = int(self.lattice_shape[0] / 2), int(self.lattice_shape[1] / 2)
                x.append(xcoord)
                y.append(ycoord)
                
                #generate 2d gaussian centered at xcoord,ycoord
                mu=np.array([xcoord,ycoord])
                Z = multivariate_gaussian(mu=mu, Sigma=np.array([[ 0.01 , 0], [0,  0.01]]),scale=self.lattice_shape)
            
                path_density = path_density+Z#np.random.multivariate_normal([xcoord,ycoord], cov = [[1,0.5],[0.5,1]])
            path_density = path_density/path_density.sum()
            plt.contourf(path_density.T)
            plt.scatter(x,y,marker="*",color="k",s=100)
            for (i,j) , (m,n) in G.edges():
                    G[(i,j)][(m,n)]["weight"] = path_density[i,j]
        return G
                
        
    def generate_path_lengths(self):
        #number of paths sampled from a predefined distribution path_distribution
        # number_of_paths = len(self.path_distribution)#int(self.lattice_shape[0]*self.lattice_shape[1])
        for i in range(len(self.path_distribution) ):
             self.path_lengths_expected.append(self.path_distribution[i])
        self.generate_sources(len(self.path_distribution))
        
        
    def generate_sources(self,number_of_paths):
        #generate sources
        degrees = dict(self.graph.degree(weight="weight"))
        degrees_multiply = []
        for key,val in degrees.items():
            for i in range(int(number_of_paths*val)):
                degrees_multiply.append(key)
        #sampling based on weighted degree, defined by path_density - high weighted degree nodes are sampled more often
        #therefore more paths originate at "core" areas.
        self.paths = [[i] for i in random.sample(degrees_multiply,number_of_paths)]
        self.sources = [i[0] for i in self.paths]
        
        #uniform sampling
        #for i in range(number_of_paths):
            #self.paths.append([(np.random.randint(0,self.lattice_shape[0]),np.random.randint(0,self.lattice_shape[1]))])
        
        
    def get_target(self,source,path_length):
        
        all_targets = []
        for n in self.graph.nodes():
            d  =self.compute_distance(source,n)
            # print(d)
            if (d >=path_length-self.unitball_delta ) & (d < self.unitball_delta+ path_length ) and n!=source:
                all_targets.append(n)
        if all_targets:
                
            return random.choice(all_targets)
        else:
            return None
        
        
    def compute_distance(self,source_coord,target_coord):
        source_coord,target_coord = np.array(source_coord)*self.length_unit,np.array(target_coord)*self.length_unit
        return (sum((source_coord- target_coord)**2))**0.5



    def generate_source(self):
        return (np.random.randint(self.lattice_shape[0]), np.random.randint(self.lattice_shape[1]))
    
    def generate_paths(self):
        
        #first, sample lengths and obtain the source based on coreness of the graph
        self.generate_path_lengths()
        
        # for i in range(len(self.paths)):
        i = 0
        while i < len(self.paths):
            #for each path, check what targets are possible that  would be approximately of a 
            #length, as was sampled from the path_distribution
            target = self.get_target(self.paths[i][0], max(900, min(self.path_lengths_expected[i], (self.lattice_shape[0]+1) * self.length_unit)))
            if target:
                temp_paths = list(nx.all_shortest_paths(self.graph,self.paths[i][0],target))
                temp_path = temp_paths[np.random.randint(0, len(temp_paths))]
                if len(temp_path) > 2:
                    self.paths[i]=temp_path
                    i+=1
                else:
                    self.paths[i][0] = self.generate_source()

            else:
                self.paths[i][0] = self.generate_source()                
            #?else perhaps remove a path if target was not found?
            
            
               


#TODO: visualise paths on a grid
#TODO: perhaps the starting points should be on the edges

def generate_flow(save_path, idx):
    # np.random.seed(2)

    grid_path = '../scenarios/5x5/roadnet.json'
    with open(grid_path, "r") as grid_file:
        data = json.load(grid_file)

    grid_intersections = [((int((x['point']['x'])/300)), int((x['point']['y'])/300), x['id']) for x in data['intersections'] if not x['virtual']]
    grid_dict = {}

    for intersection in grid_intersections:
        grid_dict.update({(intersection[0], intersection[1]) : intersection[2]})


    grid_roads = [(x['startIntersection'], x['endIntersection'], x['id']) for x in data['roads']]
    road_dict = {}

    for road in grid_roads:
        road_dict.update({(road[0], road[1]) : road[2]})

        
    tp =TrafficPaths(path_distribution=np.random.normal,number_cores=1,lattice_shape=[5,5],\
                     length_unit=300,unitball_delta=900,loc=1500,scale=100,size=3000)

    #unitball_delta - thickness of a concentric layer within which targets are allowed
    #parameters for the path distribution:
    #loc - avg. path length
    #scale - we don't know, maybe variance
    tp.generate_paths()

    

    
    inter_paths = []

    for path in tp.paths:
        new_path = []
        for point in path:
            new_path.append(grid_dict[point])
        new_path.reverse()
        inter_paths.append(new_path)



    flow_path = '../scenarios/5x5/example_flow.json'
    with open(flow_path, "r") as flow_file:
        flow = json.load(flow_file)


    new_flow = []
    for path in inter_paths:
        new_path = []
        for i in range(len(path)-1):
            new_path.append(str(road_dict[(path[i], path[i+1])]))
        flow[0]['route'] = new_path
        flow[0]['startTime'] = np.random.randint(0, 200)
        flow[0]['endTime'] = flow[0]['startTime']
        new_flow.append(copy.copy(flow[0]))

    # print(len([len(x['route']) for x in new_flow if len(x['route']) <= 1]))
    # print([len(x['route']) for x in new_flow])

    # new_flow_path = '../scenarios/5x5/flow.json'
    new_flow_path = save_path + '/flow.json'
    with open(new_flow_path, "w") as new_flow_file:
        json.dump(new_flow, new_flow_file)

        
    # print(inter_paths[0])
    # print([len(x) for x in inter_paths], len(inter_paths))
    
    # # check if sources are where on average centered correctly
    # x= tp.sources
    # xx = [i[0] for i in x]
    # plt.figure()
    # xy = [i[1] for i in x]
    # plt.hist(xx,10)
    # plt.show()
    # plt.hist(xy,10)
    # plt.show()
    
    # # check the network length of paths is approx loc
    # plt.hist([len(x)-1 for x in tp.paths])
    
    # # check euclidean lengths is approx loc
    # plt.hist([tp.compute_distance(tp.paths[i][0],tp.paths[i][-1]) for i in range(1000)])




if __name__=='__main__':

    # np.random.seed(2)

    path = '../scenarios/5x5_900_100_3k/'
    
    config = {"interval": 1, "seed": 0, "dir": path, "roadnetFile": "../roadnet.json", "flowFile": "flow.json", "rlTrafficLight": True, "saveReplay": False, "roadnetLogFile": "roadnetLogFile.json", "replayLogFile": "replayLogFile.txt", "laneChange": False}

        
    os.mkdir(path)
    for idx in range(200):
        os.mkdir(path + str(idx))
        generate_flow(path + str(idx), idx)

        config['dir'] = path + 'scenarios/' + str(idx) + '/'

        with open(path + str(idx) + '/' + str(idx) + '.config', 'w') as config_file:
            json.dump(config, config_file)
