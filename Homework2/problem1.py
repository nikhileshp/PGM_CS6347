from pgmpy.readwrite import UAIReader
import numpy as np
import operator

def swapPositions(list, pos1, pos2): 
      
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list

class GraphicalModel():
    
    def __init__(self):
        self.elim_order = []
        self.evid = {}
        self.result = 1

    def readUAI(self, path):
        reader = UAIReader(path)
        self.variables = reader.get_variables()
        self.network = reader.get_network_type()
        self.edges = reader.get_edges()
        self.tables =  reader.get_tables()
        self.domain = reader.get_domain()
        self.current_edges = list(self.edges)

        # print(self.tables)

    def print_edges(self):
        print(self.current_edges)


    def print(self):
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
        edges = []
        for i in self.tables:
            if(var in i[0]):
                edges.append(i[0])
        
        return(edges)

    def edges_of_node(self,var):
        edges = []
        for i in self.current_edges:
            if(var in i):
                edges.append(i)
        
        return(edges)

    def order(self):

        order= []
        ce = list(self.current_edges)
        while(len(self.current_edges)>0):
            degree = {}
            for i in self.current_edges:
                n1 = i[0]
                n2 = i[1]
                if n1 not in degree:
                    degree[n1]=1
                else:
                    degree[n1]+=1
                if n2 not in degree:
                    degree[n2]=1
                else:
                    degree[n2]+=1
            
            
            deg = sorted(degree.items(), key=operator.itemgetter(1))
            node = deg[0][0]
            order.append(node)
            edges = self.edges_of_node(node)
            clique = []
            for edge in edges:
                for i in edge:
                    if i not in clique:
                        clique.append(i)
            clique.remove(node)
            clique = sorted(clique)

            for i in range(len(clique)-1):
                for j in range(i+1,len(clique)):
                    tup = (clique[i],clique[j])
                    if(tup not in self.current_edges):
                        self.current_edges.append(tup)
            
            for i in edges:
                self.current_edges.remove(i)


        if(len(order)!=(len(self.domain)-len(self.evid))):
            
            x = list(set(gm.domain.keys())-set(order))
            for i in self.evid:
                x.remove("var_"+i)
            order.extend(x)
            self.elim_order = order
            
        self.current_edges = list(ce)
        # self.elim_order = ['var_1','var_2','var_3','var_6',  'var_7', 'var_8']
        print("Order of Elimination \n", self.elim_order)
    

    def instantiate(self,path):
        with open(path,"r") as f:
            x = f.read().strip().strip("\n").split(" ")
            l = x.pop(0)
            for i in range(int(l)):
                self.evid[str(x[2*i])] = int(x[2*i+1])

            
            factors = list(self.tables)
            for i in self.evid:  
                for j in factors:
                    if('var_'+i in j[0]):
                        # print('var_'+i, j[0])
                        self.new_factor(j,self.evid)

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
                    self.result = self.result * res[0]
        if(len(tup[0])>0):
            self.tables.append(tup)

    def product(self, edge1, edge2):

        common = list((set(edge1) & set(edge2)))[0]
        idx1 = edge1.index(common)
        idx2 = edge2.index(common)
        new_fn = sorted(set(edge1+edge2))

        l = 1
        for i in new_fn:
            l = l*int(self.domain[i])

        res = np.zeros(l)
        x =  "{" + ':0{}b'.format(len(new_fn)) + "}"

        for i in self.tables:
            if(i[0] == list(edge1)):
                val1 = i[1]
        self.tables.remove(tuple([list(edge1),val1]))


        for i in self.tables:
            if(i[0] == list(edge2)):
                val2 = i[1]
        self.tables.remove(tuple([list(edge2),val2]))


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
            res[i] = float(val1[idx1]) * float(val2[idx2])

        new_table_element = [new_fn,list(res)]

        self.tables.append(tuple(new_table_element))
        return new_fn



    def sumOut(self, edge, var):
        l=1
        for i in edge:
            if(i!=var):
                l = l*int(self.domain[i])
        print("Summing Out {1} from {0}".format(edge,var))
        res = np.zeros(l)
        for i in self.tables:
            if(i[0] == list(edge)):
                val = i[1]

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
                res[idx] += float(val[i])
       
        else:
            res = np.zeros(1)
            res[0] = float(val[0]) + float(val[1])

        
        if(len(res)>1):
            new_table_element = [edge,list(res)]
            # self.current_edges.append(tuple(edge))
            self.tables.append(tuple(new_table_element))

        return list(res)



    def variable_elimination(self):
        print("Graph Edges")
        print(self.current_edges)
        while(len(self.elim_order)>0):
        
            var = self.elim_order.pop(0)
            print("Current Variable ",var)
            edges = self.vars_of_edge(var)
            print("Edges containing that node: ",edges)
            while(len(edges)>=2):
                print("Product of {} {}".format(edges[0],edges[1]))
                new_factor = self.product(edges[0],edges[1])
                print("New Factor {}".format(new_factor ))
                edges = self.vars_of_edge(var)           
            if(len(edges)>=1):
                new_edge = edges[0]
                res = self.sumOut(new_edge,var)
            else:
                res = self.sumOut([var],var)
            if(len(res)==1):
                s = res[0]
                self.result = self.result * s
                print("Total: ",self.result)
            
        
        
        print("Remaining Factors: ",self.tables)
        
        print(np.log10(self.result))

if __name__=="__main__":
    gm = GraphicalModel()
    gm.readUAI('homework2_files/3.uai')
    gm.instantiate('homework2_files/3.uai.evid')
    gm.order()    
    gm.variable_elimination()

