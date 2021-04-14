
import numpy as np
import operator
import os
import random
import copy
import sys
from decimal import Decimal
from time import time
from multiprocessing import Pool, cpu_count
import functools
import statistics

class GraphicalModel():
    
    def __init__(self):
        self.elim_order = []
        
        self.evid = {}
        self.result = Decimal(1)
        self.factors = []
        self.num_vars = 0
        self.domain = {}
        self.variables = []
        self.edges = []
        self.tables = []
        self.bucket = []
        self.current_edges = []


    def readUAI(self, path):
        # Function to read the data from the UAI file
        fname = (os.path.basename(path))
        directory  = os.path.dirname(path)
        self.pr_path = directory+"/"+fname+'.PR'
        
        with open(path) as l:
            lines = l.readlines()
            l.close()

        num_vars = int(lines[1])
        self.num_vars = num_vars
        num_cliques = int(lines[3])
        domain = lines[2].split()
       

        print("Reading File")
        
        for i in range(num_vars):
            start = "var_"
            self.variables.append(start+str(i))
            self.domain[start+str(i)] = domain[i]
    
        for i in range(4, 4+num_cliques):
            l = lines[i]
            x = l.split()
            if(x[0] == "1"):
                x.pop(0)
                x[0] = "var_"+x[0]
                self.factors.append(x)
                
            else:
                x.pop(0)
                for e in range(len(x)):
                    x[e] = "var_"+x[e]
                self.factors.append(x)
                self.edges.append(tuple(x))

        self.edges = set(self.edges)
    
        for i in range(num_cliques):
            vals = (lines[5+num_cliques+3*i+1]).split()
            # for j in range(0,len(vals)):
            #     vals[j] = np.log10(float(vals[j]))
            temp = (self.factors[i],vals)
            self.tables.append(temp)
            
            # print(self.tables)
        self.current_edges =  list(self.edges)
        self.table_save = copy.deepcopy(self.tables)
        self.factor_save = copy.deepcopy(self.factors)
    
    def reset(self):
        self.tables = copy.deepcopy(self.table_save)
        self.result = Decimal(1)
        self.elim_order = copy.deepcopy(self.elim_order_save)
        self.current_edges = list(self.edges)
        self.evid = []

    def print_edges(self):
        #Function to print the remaining edges of the Graph
        print(self.current_edges)


    def print(self):
        #Function to print the remaining factors in the table
        for i in self.tables:
            k = i[0]
            v = i[1]
            var = "".join('{}\t'.format(i) for i in k)
            var = var + "    " + "Probability"
            print(var)
            x =  "{" + ':0{}b'.format(len(k)) + "}"
            l = 2**len(k)
            
            binary = [x.format(i) for i in range(l)]
            for e in range(l):
                b = list(binary[e])
                value = v[e]
                pr = ''.join('{}\t'.format(i) for i in b)
                pr = pr + '    ' + str(value)
                print(pr)
            print()



    def vars_of_edge(self,var):
        #Function to return factors that have a particular variable from the tables dictionary
        edges = []
        for i in self.tables:
            if(var in i[0]):
                edges.append(i[0])
        return(edges)



    def edges_of_node(self,var):
        #Function to return edges of the node based on the initial edges in the graph
        edges = []
        for i in self.current_edges:
            if(var in i):
                edges.append(i)
        return(edges)



    def order(self):
        # Function to find the min degree order of variable elimination
        order= [] 
        edge_matrix = np.zeros((self.num_vars,self.num_vars),dtype='int64')
        for i in list(self.edges):
            
            n1 = int(i[0].strip("var_"))
            n2 = int(i[1].strip("var_"))
            edge_matrix[n1][n2] = 1
            edge_matrix[n2][n1] = 1
        

        while(len(order)<=(self.num_vars-2)):
            # print(edge_matrix)
            best = ''
            best_count = self.num_vars+1
            for i in range(0,self.num_vars):
                bins = np.bincount(edge_matrix[i])
                if(len(bins)>1):
                    num_1s = bins[1]
                    if(num_1s< best_count):
                        best = 'var_'+str(i)
                        best_count = num_1s
                        
            # print(best, best_count)
            order.append(best)
            best_var = int(best.strip("var_"))
            res = np.where(edge_matrix[best_var] == 1)

            # print(res)
            for i in range(0,self.num_vars):
                edge_matrix[best_var][i] = 0
                edge_matrix[i][best_var] = 0
        
            
            for i in range(0,len(res[0]) -1):
                for j in range(i+1,len(res[0])):
                    n1 = res[0][i]
                    n2 = res[0][j]
                    # print(n1,n2)
                    edge_matrix[n1][n2] = 1
                    edge_matrix[n2][n1] = 1
        
        last_var = list(set(self.variables)-set(order))[0]
        order.append(last_var)

        # print(order)
        self.elim_order = order
        # print("Length")
        # print(len(self.elim_order))
        for j in self.evid:
            var = 'var_'+j
            self.elim_order.remove(var)
        self.elim_order_save = copy.deepcopy(self.elim_order)
        # print("Order of Elimination \n", self.elim_order)
        return(self.elim_order)
    

    def instantiate(self,path=None, X=None, verbose=False):
        #Function to instantiate evidence given in the .uai.evid file
        # print(self.tables)
        if(X==None and path!=None):
            
            with open(path,"r") as f:
                x = f.read().strip().strip("\n").split(" ")
                l = x.pop(0)
                for i in range(int(l)):
                    self.evid[str(x[2*i])] = int(x[2*i+1])

        elif(X!=None and path==None):
            self.evid = X
            if(verbose==True):
                print("\nSample Values")
                print(self.evid)   
            for i in X:
                var = 'var_'+i
                if(var in self.elim_order): 
                    self.elim_order.remove(var)

        
        
        else:
            print("No arguments provided")

        
        factors = list(self.tables)
        for i in self.evid:  
            for j in factors:
                if('var_'+i in j[0]):
                    # print('var_'+i, j[0])
                    # print(j,self.evid)
                    self.new_factor(j,self.evid)
                    # print(self.result)
        remove_edges = []
        for i in self.evid:
            var = 'var_'+i
            for j in self.current_edges:
                if(var in j):
                    remove_edges.append(j)

        remove = list(set(remove_edges))
        
        for i in remove:
            self.current_edges.remove(i)


    def new_factor(self, tup, evid):
        #Function to create new factors. Used in the instantiation function
        for i in evid:
            var = "var_"+i
            if(var in tup[0]):
                if(tup in self.tables):
                    self.tables.remove(tup)
                l = int(len(tup[0]))

                res = np.zeros(int(len(tup[1])/2))

                x =  "{" + ':0{}b'.format(l) + "}"
                index = tup[0].index(var)

                tup[0].pop(index)
                
                
                for j in range(len(tup[1])):    
                    binary = x.format(j)
                    list_b = list(binary)
                    b_val = list_b.pop(index)
                    if(str(evid[i])==b_val):
                        bstr = ""  
                        for ele in list_b:  
                            bstr += ele  
                        if(bstr == ""):
                            bstr = "0"
                        idx = int(bstr,2)
                        res[idx] = tup[1][j]

                if(len(tup[0])>0):
                    tup = (tup[0],list(res))
                else:
                    self.result = self.result * Decimal(res[0])
        if(len(tup[0])>0):
            self.tables.append(tup)

    def product(self, edge1, edge2,verbose=False):
        #Function to calculate product of two factors
        common = list((set(edge1) & set(edge2)))[0]
        idx1 = edge1.index(common)
        idx2 = edge2.index(common)
        new_fn = sorted(set(edge1+edge2))

        l = 1
        for i in new_fn:
            l = l*int(self.domain[i])

        res = [Decimal(0)]*l
        x =  "{" + ':0{}b'.format(len(new_fn)) + "}"

        for i in self.tables:
            if(i[0] == list(edge1)):
                val1 = i[1]
        self.tables.remove(tuple([list(edge1),val1]))


        for i in self.tables:
            if(i[0] == list(edge2)):
                val2 = i[1]
        self.tables.remove(tuple([list(edge2),val2]))

        
        if(verbose==True):
            print(val1,val2)
        for i in range(l):    
            binary = x.format(i)
            list_b = list(binary)
            val_map = dict(zip(new_fn,list_b)) 
            bstr1 = ""
            bstr2 = ""
            for j in edge1:
                bstr1 = bstr1 + val_map[j]
            for j in edge2:
                bstr2 = bstr2 + val_map[j]
            idx1 = int(bstr1,2)
            idx2 = int(bstr2,2)
            res[i] = Decimal(val1[idx1]) * Decimal(val2[idx2])

        new_table_element = [new_fn,list(res)]
        # print("New Factor :", new_table_element)
        self.tables.append(tuple(new_table_element))
        return new_fn



    def sumOut(self, edge, var,verbose=False):
        #Function to sum out a variable from a factor
        l=1
        for i in edge:
            if(i!=var):
                l = l*int(self.domain[i])
        if(verbose==True):
            print("Summing Out {1} from {0}".format(edge,var))
        res = [Decimal(0)]*l
        for i in self.tables:
            if(i[0] == list(edge)):
                val = i[1]

        if(verbose==True):
            print(val)
        x =  "{" + ':0{}b'.format(len(edge)) + "}"
        index = edge.index(var)

        self.tables.remove(tuple([list(edge),val]))
        if(len(edge)>1):
            if(tuple(edge) in self.current_edges):
                self.current_edges.remove(tuple(edge))

        edge = list(edge)
        edge.pop(index)
        if(len(val)>2):
            for i in range(len(val)):    
                binary = x.format(i)
                list_b = list(binary)
                list_b.pop(index)
                val_map = dict(zip(edge,list_b)) 
                bstr = ""
                for j in edge:
                    bstr = bstr + val_map[j]
                idx = int(bstr,2)
                res[idx] += Decimal(val[i])
       
        else:
            res = [Decimal(0)]
            res[0] = Decimal(val[0]) + Decimal(val[1])

        
        if(len(res)>1):
            new_table_element = [edge,list(res)]
            # self.current_edges.append(tuple(edge))
            self.tables.append(tuple(new_table_element))

        return list(res)


    def buckets(self, order):
        # print(self.factors)
        # print((self.factors))
        self.bucket = []
        for i in order:
            
            x = [i]
            factors = []
            for e in self.factors:
                if i in e:
                    factors.append(e)

            # print(factors)
            
            variables = []
            for e in factors:
                variables += e
                
            variables = list(set(variables))
            
            # print(variables)
            for f in factors:
                self.factors.remove(f)
            
            x.append(variables)

            self.bucket.append(tuple(x))
            new_factor = variables.copy()
            new_factor.remove(i)
            self.factors.append(new_factor)
            # print(self.factors)
        self.factors = copy.deepcopy(self.factor_save)
        # print(self.bucket)
        
    def cluster_length(self):
        max_l = 0
        for i in self.bucket:
            l = len(i[1])
            if(l>max_l):
                max_l = l
        return max_l

    def wCutset(self, w):
        X = []
        while(self.cluster_length()>w+1):
            # print()
            all_vars = []
            # print("Buckets: ",self.bucket)
            # print("Cluster Length: ",self.cluster_length())
            for i in self.bucket:
                all_vars += i[1]
            # print(all_vars)
            max_ele = max(all_vars,key=all_vars.count)
            count = all_vars.count(max_ele)
            # print("Most Occuring Ele: ",max_ele)
            # print("Count: ",count)
            for i in range(0,len(self.bucket)):
                if max_ele in self.bucket[i][1]:
                    self.bucket[i][1].remove(max_ele)
            X.append(max_ele)
            # for i in self.bucket:
            #     print(i)
        return X
            
    def generate_sample(self,X):
        samples = {}
        for i in X:
            key =i.strip("var_")
            gen = random.random()
            if(gen<0.5):
                samples[key]=0
            else:
                samples[key]=1
        return samples
    
    def sample_variable_elimination(self, X, logq, verbose=False,generate_sample=True,X_samples=None):
            
            if(generate_sample==True):
                X_samples = self.generate_sample(X)
            self.instantiate(X=X_samples)
            while(len(self.elim_order)>0):
                var = self.elim_order.pop(0)
                # print("Current Variable ",var)
                edges = self.vars_of_edge(var)
                # print("Edges containing that node: ",edges)
                while(len(edges)>=2):
                    if(verbose):
                        print("Product of {} {}".format(edges[0],edges[1]))
                    new_factor = self.product(edges[0],edges[1],verbose=verbose)
                    if(verbose):
                        print("New Factor {}".format(new_factor ))
                    edges = self.vars_of_edge(var)                  
                if(len(edges)>=1):
                    new_edge = edges[0]
                    res = self.sumOut(new_edge,var,verbose=verbose)

                else:
                    res = self.sumOut([var],var,verbose=verbose)
                if(len(res)==1):
                    s = res[0]
                    if(s>0):
                        # self.result = self.result + np.log10(s)
                        self.result = self.result * Decimal(s)
                        # print(self.tables)                            
                        # print(self.result)
                    
            if(verbose==True):    
                print("Variable Elimination Total: (Log10)",self.result)
            sample_output = float(np.log10(self.result))-logq
            if(verbose==True):
                print("Result After Subtracting Log q - ",sample_output)
            if(verbose==True):
                print()
                print("--------------------------------------------------------")
                print()
            self.reset()
            return (sample_output, X_samples)


    def main(self,w,n, verbose=True, details=False):


        n_100s = int(n/100)
        order = self.order()
        self.buckets(order)
        X = self.wCutset(w)
        if(verbose==True):
            print("Number of Variables in Cutset: {}".format(len(X)))
        q = 1
        if(verbose==True):
            print("Cutset (X): ")
            print(X)
        Q = []
        for i in X:
            if i in self.domain:
                Q.append(int(self.domain[i]))
        q = functools.reduce(operator.mul, Q)
        q = 1/q
        logq = np.log10(q)
    
        if(verbose==True):
            print("Log Q : {}".format(logq))
        results = [] 
        samples = []
        
        
        if(verbose==True):
            print("\nSampling Based Variable Elimination with w: {} and N: {} and logq: {}".format(w,n,logq))
        
            print()
        start_time = time()

        results = []
        samples = []

        args = [[X,logq,details]]*n
        # print(args)
        with Pool() as pool:
            (results, samples) = zip(*pool.starmap(self.sample_variable_elimination,args))


        
        avg1 = (sum(results)/len(results))
        time1 = time()-start_time
        if(verbose==True):
            print("(Normal Sampling) Time of Execution: {}".format(time1))
            print("Average of {} Iterations: {}".format(len(results), avg1))

        start_time = time()
        results=[]
        for i in range(0,n_100s):

            Q = []
            for ele in X:
                indexes = []
                for j in range(0,i*100):
                    if samples[j][ele.strip("var_")] == 0:
                        indexes.append(j)

                
                if results:
                    # print(indexes)
                    weights = [results[e] for e in indexes]
                    q_xi = sum(weights)/sum(results)
                    
                    Q.append(q_xi)
                    
                    q = functools.reduce(operator.mul, Q)
                    # print(i,q)
                    logq = np.log10(q)
            if(verbose==True):
                print("Iteration {} of 100s, Logq = {}, q={}".format(int(i/100)+1,logq,q))
            args = [[X,logq,details,False,samples[idx]] for idx in range(i*100,(i+1)*100)]
            with Pool() as pool:
                c_result, _ = zip(*pool.starmap(self.sample_variable_elimination,args))

            results.extend(list(c_result))
        # print(results)    
        avg2 = (sum(results)/len(results))
        time2 = time()-start_time
        if(verbose==True):
            print("(Adaptive Sampling) Time of Execution: {}".format(time2))
            print("Average of {} Iterations: {}".format(len(results), (avg2)))
        q = 1
        self.reset()
        return avg1, avg2, time1, time2
        
    def run(self, w1,w2, n, seed=1,verbose=False):
        path = self.pr_path
        # print(path)
        with open(path,"r") as f:
            x = (f.readlines())
            actual_val = float(x[1].strip("\n"))
        for i in range(w1 ,w2):
            for j in n:
                
                print("w={}, n={}".format(i,j))
                uniform_e = []
                adaptive_e = []
                uniform_t = []
                adaptive_t = []
                for idx in range(seed):
                    random.seed(idx)
                    print("Iteration {} of {}".format(idx+1,seed))
                    res1, res2, t1, t2 = gm.main(i,j,verbose=verbose)
                    e1 = float(actual_val-res1)/float(actual_val)
                    e2 = float(actual_val-res2)/float(actual_val)
                    uniform_e.append(e1)
                    adaptive_e.append(e2)
                    uniform_t.append(t1)
                    adaptive_t.append(t2)
                # print("\nFor w = {} and n = {}".format(i,j))
                print("Uniform Proposal E= {} +/- {} ".format(statistics.mean(uniform_e),statistics.stdev(uniform_e)))
                print("Uniform Proposal T= {} +/- {} ".format(statistics.mean(uniform_t),statistics.stdev(uniform_t)))
                print("Adaptive Proposal E= {} +/- {} ".format(statistics.mean(adaptive_e),statistics.stdev(adaptive_e)))
                print("Adaptive Proposal T= {} +/- {} \n".format(statistics.mean(adaptive_t),statistics.stdev(adaptive_t)))


if __name__=="__main__":
    for f in ["Grids_18.uai", ]:
        gm = GraphicalModel()
        print("Filename: {}".format(f))
        gm.readUAI('homework3_files/{}'.format(f))  #Path to the uai file 
        w1 = 3
        w2 = 6
        seed = 10
        n = [1000,]
        if(seed<2):
            print("Need atleast 2 seed to find average")
            quit()
        gm.run(w1,w2,n,seed=seed,verbose=False)
    

