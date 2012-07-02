import random
import copy
import math

#svm_type
SVM_C_SVC = 0
SVM_NU_SVC = 1
SVM_ONE_CLASS = 2
SVM_EPSILON_SVR = 3
SVM_NU_SVR = 4

#kernel_type
SVM_LINEAR = 0
SVM_POLY = 1
SVM_RBF = 2
SVM_SIGMOID = 3
SVM_PRECOMPUTED = 4

INF = float('INF')
LOWER_BOUND = 0
UPPER_BOUND = 1
FREE = 2

LIBSVM_VERSION = 310
rand = random.random

svm_type_table = ( "c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr" )
kernel_type_table = ( "linear", "polynomial", "rbf", "sigmoid", "precomputed" )

def a2d(n1,n2,value):
    tmp = []
    for i in xrange(0, n1):
        tmp.append([value for j in xrange(0, n2)])
    return tmp

class svm_node:
    index = 0 #int
    value = 0. #double
    
    def __init__(self, **kwargs):
        if 'index' in kwargs: self.index = kwargs['index']
        if 'value' in kwargs: self.value = kwargs['value']
        
    def __repr__(self):
        return "svm_node<%s:%s>" % (self.index, self.value)

class svm_parameter:
    svm_type = SVM_C_SVC
    kernel_type = SVM_RBF
    degree = 3 #int
    gamma = 0.5 #double (default 1/num_features)
    coef0 = 0.
    cache_size = 100    #in MB
    eps = 1e-3
    C = 1   #cost
    
    nr_weight = 0 #int
    weight_label = None #int[]
    weight = None #double[]
    
    nu = 0.5 #double for NU_SVC, ONE_CLASS, and NU_SVR
    p = 0.1 #for EPSILON_SVR
    
    #int
    shrinking = 1
    probability = 0
    
    def __init__(self, **kwargs):
        if 'svm_type' in kwargs: self.svm_type = kwargs['svm_type']
        if 'kernel_type' in kwargs: self.kernel_type = kwargs['kernel_type']
        if 'degree' in kwargs: self.degree = kwargs['degree']
        if 'gamma' in kwargs: self.gamma = kwargs['gamma']
        if 'coef0' in kwargs: self.coef0 = kwargs['coef0']
        if 'cache_size' in kwargs: self.cache_size = kwargs['cache_size']
        if 'eps' in kwargs: self.eps = kwargs['eps']
        if 'C' in kwargs: self.C = kwargs['C']
        if 'nu' in kwargs: self.nu = kwargs['nu']
        if 'p' in kwargs: self.p = kwargs['p']
        
    def __repr__(self):
        return "<svm_parameter(svm_type=%s,kernel_type=%s)>" % (svm_type_table[self.svm_type], kernel_type_table[self.kernel_type])
    
    def clone(self):
        return copy.deepcopy(self)

class svm_problem:
    l = 0 #int
    y = None #double[]
    x = None #svm_node[][]
    
    def __init__(self,**kwargs):
        if 'y' in kwargs: self.y = kwargs['y']
        if 'x' in kwargs: self.x = kwargs['x']
        if self.y is not None: self.l = len(self.y)
    
    def __repr__(self):
        return "<y:%s, x:%s>" % (self.y, self.x)

class svm_model:
    param = None #svm_parameter
    nr_class = 0 #int
    l = None
    nSV = 0
    SV = None #svm_node[][]
    sv_coef = None #double[][]
    rho = None #double[]
    probA = None
    probB = None
    label = None #int[]
    
    def get_type(self): return self.param.svm_type
    
    def get_svr_prob(self):
        if (self.param.svm_type in [SVM_EPSILON_SVR, SVM_NU_SVR]) and self.probA != None: 
            return self.probA[0]
        else:
            info("Model doesn't contain information for SVR probability inference")
            return 0.
            
    def get_labels(self):
        label = None
        if self.label != None:
            label = []
            for i in xrange(0, self.nr_class):
                label.append(self.label[i])
        return label
        
    def check_probability(model):
        if ((model.param.svm_type in [SVM_C_SVC, SVM_NU_SVC]) and model.probA != None and model.probB != None) \
        or ((model.param.svm_type in [SVM_EPSILON_SVR, SVM_NU_SVR]) and model.probA != None): 
            return 1
        else:
            return 0
            
    def save(model, filename):
        fp = open(filename, "w")
        param = model.param
        fp.write("svm_type %s\n" % (svm_type_table[param.svm_type]))
        fp.write("kernel_type %s\n" % (kernel_type_table[param.kernel_type]))
        if param.kernel_type == SVM_POLY: 
            fp.write("degree %s\n" % (param.degree))
        if param.kernel_type in [SVM_POLY, SVM_RBF, SVM_SIGMOID]: 
            fp.write("gamma %s\n" % param.gamma)
        if param.kernel_type in [SVM_POLY, SVM_SIGMOID]: 
            fp.write("coef0 %s\n" % param.coef0)
        nr_class = model.nr_class
        l = model.l
        fp.write("nr_class %s\n" % nr_class)
        fp.write("total_sv %s\n" % l)
        fp.write("rho")
        for i in xrange(0, nr_class * (nr_class-1) / 2 ):
            fp.write(" " + str(model.rho[i]))
        fp.write("\n")

        if model.label != None: 
            fp.write("label")
            for i in xrange(0, nr_class):
                fp.write(" " + str(model.label[i]))
            fp.write("\n")

        if model.probA != None: 
            fp.write("probA")
            for i in xrange(0, nr_class * (nr_class-1) / 2 ):
                fp.write(" " + str(model.probA[i]))
            fp.write("\n")

        if model.probB != None: 
            fp.write("probB")
            for i in xrange(0, nr_class * (nr_class-1) / 2 ):
                fp.write(" " + str(model.probB[i]))
            fp.write("\n")

        if model.nSV != None: 
            fp.write("nr_sv")
            for i in xrange(0, nr_class):
                fp.write(" " + str(model.nSV[i]))
            fp.write("\n")

        fp.write("SV\n")
        sv_coef = model.sv_coef # double[][]
        SV = model.SV # svm_node[][]
        for i in xrange(0, l):
            for j in xrange( 0,  nr_class - 1 ): 
                fp.write("%f " % (sv_coef[j][i]))
                
            p = SV[i]
            if param.kernel_type == SVM_PRECOMPUTED: 
                fp.write("0:" + str(int(p[0].value)) )
            else:
                for j in xrange( 0, len(p) ): 
                    fp.write(str(p[j].index) + ":" + str(p[j].value) + " ")
            fp.write("\n")
        fp.close()

    def load(model, filename):
        fp = open(filename)
        param = svm_parameter()
        model.param = param
        model.rho = None
        model.probA = None
        model.probB = None
        model.label = None
        model.nSV = None
        while True:
            cmd = fp.readline().strip()
            if len(cmd)==0: continue
            arg = ''
            if ' ' in cmd:
                arg = cmd[cmd.index(' ') + 1:]
            if cmd.startswith("svm_type"): 
                #print repr(arg), svm_type_table.index(arg)
                if arg in svm_type_table:
                    param.svm_type = svm_type_table.index(arg)
                else:
                    print "unknown svm type.\n"
                    return None
            elif cmd.startswith("kernel_type"): 
                if arg in kernel_type_table:
                    param.kernel_type = kernel_type_table.index(arg)
                else: 
                    print "unknown kernel function.\n"
                    return None
            elif cmd.startswith("degree"): 
                param.degree = int(arg)
            elif cmd.startswith("gamma"): 
                param.gamma = float(arg)
            elif cmd.startswith("coef0"): 
                param.coef0 = float(arg)
            elif cmd.startswith("nr_class"): 
                model.nr_class = int(arg)
            elif cmd.startswith("total_sv"): 
                model.l = int(arg)
            elif cmd.startswith("rho"): 
                n = model.nr_class * (model.nr_class - 1) / 2
                model.rho = [0.]*n
                st = arg.split(" ")
                for i in xrange(0, n):
                    model.rho[i] = float(st[i])
            elif cmd.startswith("label"): 
                model.label = []
                st = arg.split(' ')
                for i in xrange(0, model.nr_class): 
                    model.label.append(int(st[i]))
            elif cmd.startswith("probA"): 
                n = model.nr_class * (model.nr_class - 1) / 2
                model.probA = [0.]*n
                st = arg.split(" ")
                for i in xrange(0, n):
                    model.probA[i] = float(st[i])
            elif cmd.startswith("probB"): 
                n = model.nr_class * (model.nr_class - 1) / 2
                model.probB = [0.]*n
                st = arg.split(" ")
                for i in xrange(0, n):
                    model.probB[i] = float(st[i])
            elif cmd.startswith("nr_sv"): 
                # int
                n = model.nr_class
                model.nSV = [0]*n
                st = arg.split(" ")
                for i in xrange(0, n): 
                    model.nSV[i] = int(st[i])
            elif cmd.startswith("SV"): 
                break
            else:
                print "unknown text in model file: [" + cmd + "]\n"
                return None

        model.sv_coef = a2d(model.nr_class - 1, model.l, 0.)
        model.SV = [[]] * model.l
        for i in xrange(0, model.l): 
            line = fp.readline().strip()
            st = line.split(" ")
            for k in xrange(0, model.nr_class - 1): 
                model.sv_coef[k][i] = float(st[k])

            model.SV[i] = []
            
            t = model.nr_class-1
            for j in xrange(0, len(st)-model.nr_class+1): 
                pair = st[t].split(':')
                model.SV[i].append( svm_node(index=int(pair[0]), value=float(pair[1]) ) )
                t += 1
        fp.close()
        return model

class Cache:
    class head_t:
        prev = None #head_t
        next = None #head_t
        data = None #float[]
        len = 0 #int
        
    l = None #int
    size = None #long
    head = None #head_t[]
    lru_head = None #head_t

    def __init__(self, l_, size_):
        """
        Parameters:
            l_: int 
            size_: long 
        """
        self.l = l_
        self.size = size_
        self.head = []
        for i in xrange(0, self.l): self.head += [Cache.head_t()]
        self.size /= 4
        self.size -= self.l * (16 / 4)
        self.size = max(self.size, 2 * long(self.l) )
        self.lru_head = Cache.head_t()
        self.lru_head.next = self.lru_head.prev = self.lru_head

    def lru_delete(self, h):
        h.prev.next = h.next
        h.next.prev = h.prev

    def lru_insert(self, h):
        h.next = self.lru_head
        h.prev = self.lru_head.prev
        h.prev.next = h
        h.next.prev = h

    def get_data(self, index, data, len_):
        """
        Returns int
        Parameters:
            index: int
            data: float[][]
            len: int
        """
        # head_t
        h = self.head[index]
        if h.len > 0: self.lru_delete(h)
        
        more = len_ - h.len
        if more > 0: 
            while self.size < more:
                # head_t
                old = lru_head.next
                self.lru_delete(old)
                self.size += old.len
                old.data = None
                old.len = 0

            new_data = [0.]*len_
            if h.data is not None: 
                new_data = copy.deepcopy(h.data)
                #System.arraycopy(h.data, 0, new_data, 0, h.len)
            h.data = new_data
            self.size -= more
            h.len,len_ = len_,h.len

        self.lru_insert(h)
        data[0] = h.data
        return len_

    def swap_index(self, i, j):
        if i == j: return
        if self.head[i].len > 0: 
            self.lru_delete(self.head[i])
        if self.head[j].len > 0: 
            self.lru_delete(self.head[j])
        
        #swap
        self.head[j].data,self.head[i].data = self.head[i].data,self.head[j].data
        self.head[i].len,self.head[j].len = self.head[j].len,self.head[i].len
        
        if self.head[i].len > 0: self.lru_insert(head[i])
        if self.head[j].len > 0: self.lru_insert(head[j])
        if i > j: i,j = j,i
        
        h = self.lru_head.next
        while h != self.lru_head : 
            if h.len > i: 
                if h.len > j:
                    h.data[j], h.data[i] = h.data[i], h.data[j]
                else:
                    self.lru_delete(h)
                    self.size += h.len
                    h.data = None
                    h.len = 0
            h = h.next

class QMatrix:
    def get_Q(self, column, len):
        """
        Returns float[]
        Parameters:
            column: int     
            len: int
        """
        pass

    def get_QD(self):
        """
        Returns double[]
        """
        pass

    def swap_index(self, i, j):
        pass

class Kernel(QMatrix):
    def __init__(self, l, x_, param):
        """
        Parameters:
            l: int
            x_: svm_node[][]
            param : svm_parameter
        """
        self.kernel_type = param.kernel_type
        self.degree = param.degree
        self.gamma = param.gamma
        self.coef0 = param.coef0
        self.x = copy.deepcopy(x_)
        if self.kernel_type == SVM_RBF: 
            self.x_square = [0.]*l
            for i in xrange( 0, l ): 
                self.x_square[i] = self.dot(self.x[i], self.x[i])
        else:
            self.x_square = None
            
    def __repr__(self):
        return "<%s_Kernel>" % (kernel_type_table[self.kernel_type])

    def swap_index(self, i, j):
        self.x[j], self.x[i] = self.x[i], self.x[j]
        if self.x_square is not None: 
            self.x_square[j], self.x_square[i] = self.x_square[i], self.x_square[j]

    @staticmethod
    def powi(base, times):
        tmp = base
        ret = 1.0
        t = times
        while (t > 0): 
            if t % 2 == 1: 
                ret *= tmp
            tmp = tmp * tmp
            t /= 2
        return ret

    def kernel_function(self, i, j):
        """
        Returns double
        Parameters:
            i: int
            j: int
        """
        if self.kernel_type == SVM_LINEAR:
            return Kernel.dot( self.x[i], self.x[j] )
        elif self.kernel_type == SVM_POLY:
            return Kernel.powi( self.gamma * Kernel.dot(self.x[i], self.x[j]) + self.coef0, self.degree )
        elif self.kernel_type == SVM_RBF:
            return math.exp( -self.gamma * (self.x_square[i] + self.x_square[j] - 2. * Kernel.dot(self.x[i], self.x[j])) )
        elif self.kernel_type == SVM_SIGMOID:
            return math.tanh(self.gamma * Kernel.dot(self.x[i], self.x[j]) + self.coef0)
        elif self.kernel_type == SVM_PRECOMPUTED:
            return self.x[i][ int(self.x[j][0].value) ].value
        else:
            return 0
            
    @staticmethod
    def dot(x, y):
        """
        Returns double
        Parameters:
            x: svm_node[]   
            y: svm_node[]
        """
        sum = 0.
        i = 0
        j = 0
        while i < len(x) and j < len(y):
            if x[i].index == y[j].index: 
                sum += x[i].value * y[j].value
                i += 1
                j += 1
            else:
                if x[i].index > y[j].index: 
                    j += 1
                else:
                    i += 1
        return sum

    @staticmethod
    def k_function(x, y, param):
        """
        Returns double
        Parameters:
            x: svm_node[]
            y: svm_node[]
            param: svm_parameter
        Java modifiers:
             static
        """
        if param.kernel_type == SVM_LINEAR:
            return Kernel.dot(x, y)
        elif param.kernel_type == SVM_POLY:
            return Kernel.powi(param.gamma * Kernel.dot(x, y) + param.coef0, param.degree)
        elif param.kernel_type == SVM_RBF:
            sum = 0.
            i,j = 0,0
            while i < len(x) and j < len(y):
                if x[i].index == y[j].index: 
                    d = x[i].value - y[j].value
                    sum += d * d
                    i += 1
                    j += 1
                else:
                    if x[i].index > y[j].index: 
                        sum += y[j].value * y[j].value
                        j += 1
                    else:
                        sum += x[i].value * x[i].value
                        i += 1

            while i < len(x):
                sum += x[i].value * x[i].value
                i += 1
            while j < len(y):
                sum += y[j].value * y[j].value
                j += 1

            return math.exp(-param.gamma * sum)

        if param.kernel_type == SVM_SIGMOID:
            return math.tanh(param.gamma * Kernel.dot(x, y) + param.coef0)
        if param.kernel_type == SVM_PRECOMPUTED:
            return x[ int(y[0].value) ].value
        else:
            return 0

class Solver:
    active_size = 0
    y = None #byte[]
    G = None #double[]

    alpha_status = None #byte[]
    alpha = None #double[]
    Q = None #QMatrix
    QD = None #double[]
    
    eps = 0. #double
    Cp = None
    Cn = None

    p = None #double[]
    active_set = None #int[]
    G_bar = None #double[]
    
    l = 0 #int
    unshrink = False #boolean

    def get_C(self, i):
        """
        Returns double
        Parameters:
            i: int
        """
        if self.y[i] > 0:
            return self.Cp
        else:
            return self.Cn

    def update_alpha_status(self, i):
        """
        Parameters:
            i: int
        """
        if self.alpha[i] >= self.get_C(i): 
            self.alpha_status[i] = UPPER_BOUND
        else:
            if self.alpha[i] <= 0.: 
                self.alpha_status[i] = LOWER_BOUND
            else:
                self.alpha_status[i] = FREE

    def is_upper_bound(self, i):
        return self.alpha_status[i] == UPPER_BOUND

    def is_lower_bound(self, i):
        return self.alpha_status[i] == LOWER_BOUND

    def is_free(self, i):
        return self.alpha_status[i] == FREE

    #static
    class SolutionInfo:
        #double
        obj = None
        rho = None
        upper_bound_p = None
        upper_bound_n = None
        r = None

    def swap_index(self, i, j):
        self.Q.swap_index(i, j)
        self.y[j],self.y[i] = self.y[i],self.y[j]
        self.G[i],self.G[j] = self.G[j],self.G[i]
        self.alpha_status[i],self.alpha_status[j] = self.alpha_status[j],self.alpha_status[i]
        self.alpha[i],self.alpha[j] = self.alpha[j],self.alpha[i]
        self.p[i],self.p[j] = self.p[j],self.p[i]
        self.active_set[i],self.active_set[j] = self.active_set[j],self.active_set[i]
        self.G_bar[i],self.G_bar[j] = self.G_bar[j], self.G_bar[i]

    def reconstruct_gradient(self):
        if self.active_size == self.l: return

        nr_free = 0
        for j in xrange( self.active_size, self.l ): 
            self.G[j] = self.G_bar[j] + self.p[j]
        for j in xrange( 0, active_size ) : 
            if self.is_free(j): 
                nr_free += 1
        if 2 * nr_free < self.active_size: 
            info("\nWarning: using -h 0 may be faster")
        if nr_free * self.l > 2 * self.active_size * (self.l - self.active_size): 
            for i in xrange( active_size, self.l ) : 
                # float[]
                self.Q_i = self.Q.get_Q(i, self.active_size)
                for j in xrange( 0, self.active_size) : 
                    if self.is_free(j): 
                        self.G[i] += self.alpha[j] * self.Q_i[j]
        else:
            for i in xrange( 0, self.active_size) : 
                if is_free(i): 
                    # float[]
                    self.Q_i = self.Q.get_Q(i, self.l)
                # double
                alpha_i = alpha[i]
                for j in xrange( active_size, self.l ) : 
                    G[j] += alpha_i * Q_i[j]

    def solve(self, l, Q, p_, y_, alpha_, Cp, Cn, eps, si, shrinking):
        """
        Parameters:
            l: int  Q: QMatrix    p_: double[]    y_: byte[]  alpha_: double[]  Cp: double    Cn: double  eps: double   si: SolutionInfo   shrinking: int
        """
        self.l = l
        self.Q = Q
        self.QD = Q.get_QD()
        self.p = copy.deepcopy(p_) 
        self.y = copy.deepcopy(y_) 
        self.alpha = copy.deepcopy(alpha_) 
        self.Cp = Cp
        self.Cn = Cn
        self.eps = eps
        self.unshrink = False
        self.alpha_status = [0]*l
        for i in xrange( 0, self.l ): 
            self.update_alpha_status(i)

        self.active_set = [i for i in xrange(0, l)]
        self.active_size = l

        self.G = [self.p[i] for i in xrange(0, l)]
        self.G_bar = [0.]*l

        for i in xrange( 0, l ) : 
            if not self.is_lower_bound(i): 
                Q_i = self.Q.get_Q(i, l)
                alpha_i = self.alpha[i]

                for j in xrange( 0, l ) : 
                    self.G[j] += alpha_i * Q_i[j]
                    
                if self.is_upper_bound(i): 
                    for j in xrange( 0, self.l ) : 
                        self.G_bar[j] += self.get_C(i) * Q_i[j]

        iter = 0
        counter = min(l, 1000) + 1
        
        working_set = [0,0] #new int[2]
        while True:
            counter -= 1
            if counter == 0: 
                counter = min(self.l, 1000)
                if self.shrinking != 0: 
                    self.do_shrinking()
                info(".")

            if self.select_working_set(working_set) != 0: 
                self.reconstruct_gradient()
                self.active_size = l
                info("*")
                if self.select_working_set(working_set) != 0: 
                    break
                else:
                    counter = 1

            i = working_set[0]
            j = working_set[1]
            
            iter += 1
            # float[]
            Q_i = self.Q.get_Q(i, self.active_size)
            Q_j = self.Q.get_Q(j, self.active_size)
            
            # double
            C_i = self.get_C(i)
            C_j = self.get_C(j)
            
            old_alpha_i = self.alpha[i]
            old_alpha_j = self.alpha[j]
            
            if self.y[i] != self.y[j]:                 
                quad_coef = self.QD[i] + self.QD[j] + 2. * Q_i[j]
                if quad_coef <= 0: 
                    quad_coef = 1.0E-12
                delta = (-self.G[i] - self.G[j]) / quad_coef
                diff = self.alpha[i] - self.alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta
                if diff > 0: 
                    if self.alpha[j] < 0: 
                        self.alpha[j] = 0.
                        self.alpha[i] = diff
                else:
                    if self.alpha[i] < 0: 
                        self.alpha[i] = 0.
                        self.alpha[j] = -diff

                if diff > C_i - C_j: 
                    if self.alpha[i] > C_i: 
                        self.alpha[i] = C_i
                        self.alpha[j] = C_i - diff
                else:
                    if self.alpha[j] > C_j: 
                        self.alpha[j] = C_j
                        self.alpha[i] = C_j + diff
            else:
                quad_coef = self.QD[i] + self.QD[j] - 2. * Q_i[j]
                if quad_coef <= 0: 
                    quad_coef = 1.0E-12
                delta = (self.G[i] - self.G[j]) / quad_coef
                sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta
                if sum > C_i: 
                    if self.alpha[i] > C_i: 
                        self.alpha[i] = C_i
                        self.alpha[j] = sum - C_i
                else:
                    if self.alpha[j] < 0: 
                        self.alpha[j] = 0
                        self.alpha[i] = sum

                if sum > C_j: 
                    if self.alpha[j] > C_j: 
                        self.alpha[j] = C_j
                        self.alpha[i] = sum - C_j
                else:
                    if self.alpha[i] < 0: 
                        self.alpha[i] = 0
                        self.alpha[j] = sum

            delta_alpha_i = self.alpha[i] - old_alpha_i
            delta_alpha_j = self.alpha[j] - old_alpha_j
            for k in xrange(0, self.active_size): 
                self.G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j

                ui = self.is_upper_bound(i)
                uj = self.is_upper_bound(j)
                
                self.update_alpha_status(i)
                self.update_alpha_status(j)

                if ui != self.is_upper_bound(i): 
                    Q_i = self.Q.get_Q(i, self.l)
                    if ui: 
                        for k in xrange( 0, self.l ) : 
                            self.G_bar[k] -= C_i * Q_i[k]
                    else:
                        for k in xrange( 0, self.l) : 
                            self.G_bar[k] += C_i * Q_i[k]

                if uj != self.is_upper_bound(j): 
                    Q_j = self.Q.get_Q(j, self.l)
                    if uj: 
                        for k in xrange( 0, self.l ) : 
                            self.G_bar[k] -= C_j * Q_j[k]
                    else:
                        for k in xrange( 0, self.l ) : 
                            self.G_bar[k] += C_j * Q_j[k]

        si.rho = self.calculate_rho()
        v = 0.
        for i in xrange( 0, self.l ) : 
            v += self.alpha[i] * (self.G[i] + self.p[i])
        si.obj = v * .5

        for i in xrange( 0, self.l ): 
            alpha_[self.active_set[i]] = self.alpha[i]

        si.upper_bound_p = Cp
        si.upper_bound_n = Cn
        info("\noptimization finished, #iter = " + str(iter) + "")

    def select_working_set(self, working_set):
        """
        Returns int
        Parameters:
            working_set: int[]
        """
        Gmax = -INF
        Gmax2 = -INF
        Gmax_idx = -1
        Gmin_idx = -1
        obj_diff_min = INF
        for t in xrange( 0, self.active_size ): 
            if self.y[t] == +1: 
                if not self.is_upper_bound(t): 
                    if -self.G[t] >= Gmax: 
                        Gmax = -self.G[t]
                        Gmax_idx = t
            else:
                if not self.is_lower_bound(t): 
                    if self.G[t] >= Gmax: 
                        Gmax = self.G[t]
                        Gmax_idx = t

        i = Gmax_idx
        Q_i = None # float[]
        if i != -1: 
            Q_i = self.Q.get_Q(i, self.active_size)
        #print "Solver.select_working_set", len(Q_i), i, self.active_size, Q_i
        for j in xrange( 0, self.active_size ) : 
            if self.y[j] == +1: 
                if not self.is_lower_bound(j): 
                    grad_diff = Gmax + self.G[j]
                    if self.G[j] >= Gmax2: 
                        Gmax2 = self.G[j]
                    if grad_diff > 0: 
                        quad_coef = self.QD[i] + self.QD[j] - 2. * self.y[i] * Q_i[j]
                        if quad_coef > 0: 
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1.0E-12
                        if obj_diff <= obj_diff_min: 
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j): 
                    grad_diff = Gmax - self.G[j]
                    if -self.G[j] >= Gmax2: 
                        Gmax2 = -self.G[j]
                    if grad_diff > 0: 
                        #print i, j, len(self.QD), len(self.y), len(Q_i)
                        quad_coef = self.QD[i] + self.QD[j] + 2.0 * self.y[i] * Q_i[j]
                        if quad_coef > 0: 
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1.0E-12
                        if obj_diff <= obj_diff_min: 
                            Gmin_idx = j
                            obj_diff_min = obj_diff

        if Gmax + Gmax2 < self.eps: 
            return 1
        working_set[0] = Gmax_idx
        working_set[1] = Gmin_idx
        return 0

    def be_shrunk(self, i, Gmax1, Gmax2):
        """
        Returns boolean
        Parameters:
            i: int
            Gmax1: double
            Gmax2: double
        """
        if self.is_upper_bound(i): 
            if self.y[i] == +1: 
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax2
        elif self.is_lower_bound(i): 
            if self.y[i] == +1: 
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax1
        else:
            return False

    def do_shrinking(self):
        Gmax1 = -INF
        Gmax2 = -INF
        for i in xrange( 0, self.active_size ) : 
            if self.y[i] == +1: 
                if not self.is_upper_bound(i): 
                    if -self.G[i] >= Gmax1: 
                        Gmax1 = -self.G[i]

                if not self.is_lower_bound(i): 
                    if self.G[i] >= Gmax2: 
                        Gmax2 = self.G[i]
            else:
                if not self.is_upper_bound(i): 
                    if -self.G[i] >= Gmax2: 
                        Gmax2 = -self.G[i]

                if not self.is_lower_bound(i): 
                    if self.G[i] >= Gmax1: 
                        Gmax1 = self.G[i]

        if not self.unshrink and Gmax1 + Gmax2 <= self.eps * 10: 
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l

        for i in xrange( 0, self.active_size ) : 
            if self.be_shrunk(i, Gmax1, Gmax2): 
                active_size -= 1
                while active_size > i:
                    if not self.be_shrunk(active_size, Gmax1, Gmax2): 
                        self.swap_index(i, active_size)
                        break

                    self.active_size -= 1

    def calculate_rho(self):
        """
        Returns double
        """
        nr_free = 0
        ub = INF
        lb = -INF
        sum_free = 0.
        for i in xrange( 0, self.active_size ): 
            yG = self.y[i] * self.G[i]
            if self.is_lower_bound(i): 
                if self.y[i] > 0: 
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            else:
                if self.is_upper_bound(i): 
                    if self.y[i] < 0: 
                        ub = min(ub, yG)
                    else:
                        lb = max(lb, yG)
                else:
                    nr_free += 1
                    sum_free += yG

        if nr_free > 0: 
            r = sum_free / nr_free
        else:
            r = (ub + lb) * .5
        return r

class Solver_NU(Solver):
    #SolutionInfo
    si = None

    def solve(self, l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking):
        self.si = si
        #super(Solver_NU, self).solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking)
        Solver.solve(self, l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking)

    def select_working_set(self, working_set):
        """
        Returns int
        Parameters:
            working_set: int[]
        """
        Gmaxp = -INF
        Gmaxp2 = -INF
        Gmaxp_idx = -1
        Gmaxn = -INF
        Gmaxn2 = -INF
        Gmaxn_idx = -1
        Gmin_idx = -1
        obj_diff_min = INF
        
        for t in xrange( 0, self.active_size ): 
            if self.y[t] == +1: 
                if not self.is_upper_bound(t): 
                    if -self.G[t] >= Gmaxp: 
                        Gmaxp = -self.G[t]
                        Gmaxp_idx = t

            else:
                if not self.is_lower_bound(t): 
                    if self.G[t] >= Gmaxn: 
                        Gmaxn = self.G[t]
                        Gmaxn_idx = t
        
        ip = Gmaxp_idx
        ind = Gmaxn_idx
        Q_ip = None #float[]
        Q_in = None
        if ip != -1: 
            Q_ip = self.Q.get_Q(ip, self.active_size)
        if ind != -1: 
            Q_in = self.Q.get_Q(ind, self.active_size)
        for j in xrange( 0, self.active_size ): 
            if self.y[j] == +1: 
                if not self.is_lower_bound(j): 
                    grad_diff = Gmaxp + self.G[j]
                    if self.G[j] >= Gmaxp2: 
                        Gmaxp2 = self.G[j]
                    if grad_diff > 0: 
                        quad_coef = self.QD[ip] + self.QD[j] - 2. * Q_ip[j]
                        if quad_coef > 0: 
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1.0E-12
                        if obj_diff <= obj_diff_min: 
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j): 
                    grad_diff = Gmaxn - self.G[j]
                    if -self.G[j] >= Gmaxn2: 
                        Gmaxn2 = -self.G[j]
                    if grad_diff > 0: 
                        quad_coef = self.QD[ind] + self.QD[j] - 2 * Q_in[j]
                        if quad_coef > 0: 
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1.0E-12
                        if obj_diff <= obj_diff_min: 
                            Gmin_idx = j
                            obj_diff_min = obj_diff

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps: 
            return 1
        if self.y[Gmin_idx] == +1: 
            working_set[0] = Gmaxp_idx
        else:
            working_set[0] = Gmaxn_idx
        working_set[1] = Gmin_idx
        return 0


    def be_shrunk(self, i, Gmax1, Gmax2, Gmax3, Gmax4):
        """
        Returns boolean
        Parameters:
            i: int  
            Gmax1: double 
            Gmax2: double    
            Gmax3: double  
            Gmax4: double
        """
        if is_upper_bound(i): 
            if y[i] == +1: 
                return (-G[i] > Gmax1)
            else:
                return (-G[i] > Gmax4)
        else:
            if is_lower_bound(i): 
                if y[i] == +1: 
                    return (G[i] > Gmax2)
                else:
                    return (G[i] > Gmax3)
            else:
                return False


    def do_shrinking(self):
        Gmax1 = -INF
        Gmax2 = -INF
        Gmax3 = -INF
        Gmax4 = -INF

        for i in xrange(0, active_size):
            if not is_upper_bound(i): 
                if y[i] == +1: 
                    if -G[i] > Gmax1: 
                        Gmax1 = -G[i]
                else:
                    if -G[i] > Gmax4: 
                        Gmax4 = -G[i]

            if not is_lower_bound(i): 
                if y[i] == +1: 
                    if G[i] > Gmax2: 
                        Gmax2 = G[i]
                else:
                    if G[i] > Gmax3: 
                        Gmax3 = G[i]


        if not unshrink and max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= self.eps * 10: 
            unshrink = True
            reconstruct_gradient()
            active_size = l

        for i in xrange(0, active_size):
            if be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4): 
                active_size -= 1
                while active_size > i:
                    if not be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4): 
                        swap_index(i, active_size)
                        break

                    active_size -= 1

    def calculate_rho(self):
        """
        Returns double
        """
        nr_free1 = 0
        nr_free2 = 0
        ub1 = INF
        ub2 = INF
        lb1 = -INF
        lb2 = -INF
        sum_free1 = 0
        sum_free2 = 0
        for i in xrange(0, self.active_size):
            if self.y[i] == +1: 
                if self.is_lower_bound(i): 
                    ub1 = min(ub1, self.G[i])
                else:
                    if self.is_upper_bound(i): 
                        lb1 = max(lb1, self.G[i])
                    else:
                        nr_free1 += 1
                        sum_free1 += self.G[i]

            else:
                if self.is_lower_bound(i): 
                    ub2 = min(ub2, self.G[i])
                else:
                    if self.is_upper_bound(i): 
                        lb2 = max(lb2, self.G[i])
                    else:
                        nr_free2 += 1
                        sum_free2 += self.G[i]

        r1 = 0.
        r2 = 0.

        if nr_free1 > 0: 
            r1 = sum_free1 / nr_free1
        else:
            r1 = (ub1 + lb1) / 2
        if nr_free2 > 0: 
            r2 = sum_free2 / nr_free2
        else:
            r2 = (ub2 + lb2) / 2
        self.si.r = (r1 + r2) / 2
        return (r1 - r2) / 2

class SVC_Q(Kernel):
    y = None #byte[]
    cache = None #Cache
    QD = None #double[]

    def __init__(self, prob, param, y_):
        """
        Parameters:
            prob: svm_problem
            param: svm_parameter
            y_: byte[]
        """
        #super(SVC_Q, self).__init__(prob.l, prob.x, param)
        Kernel.__init__(self, prob.l, prob.x, param)
        self.y = copy.deepcopy(y_)
        self.cache = Cache(prob.l, long(param.cache_size * (1 << 20)) )
        self.QD = [0.] * prob.l
        for i in xrange(0, prob.l):
            self.QD[i] = self.kernel_function(i, i)

    def get_Q(self, i, len):
        """
        Returns float[]
        Parameters:
            i: int  
            len: int
        """
        data = [[0.]]
        start = self.cache.get_data(i, data, len)

        if start < len: 
            for j in xrange( start, len) : 
                data[0][j] += float(self.y[i] * self.y[j] * self.kernel_function(i, j)) 

        return data[0]

    def get_QD(self):
        return self.QD

    def swap_index(self, i, j):
        self.cache.swap_index(i, j)
        super(SVC_Q, self).swap_index(i, j)
        self.y[j],self.y[i] = self.y[i],self.y[j]
        self.QD[j],self.QD[i] = self.QD[i],self.QD[j]

class ONE_CLASS_Q(Kernel):
    cache = None #Cache
    QD = None #double[]
    
    def __init__(self, prob, param):
        """
        Parameters:
            svm_problem prob
            svm_parameter param
        """ 
        Kernel.__init__(self, prob.l, prob.x, param)
        #super(prob.l, prob.x, param)
        self.cache = Cache(prob.l, long(param.cache_size * (1 << 20)) )
        self.QD = [0.] * prob.l
        for i in xrange(0, prob.l) : 
            self.QD[i] = self.kernel_function(i, i)

    def get_Q(self, i, len_):
        """
        Returns float[]
        Parameters:
            i: int  
            len_: int
        """
        tmp = [0.] * len_ # float[][]
        data = [tmp] #new float[1][]
        start = self.cache.get_data(i, data, len_)
        if start < len_: 
            for j in xrange(start, len_):
                data[0][j] = float(self.kernel_function(i, j)) 

        return data[0]

    def get_QD(self):
        return self.QD

    def swap_index(self, i, j):
        cache.swap_index(i, j)
        super(Kernel, self).swap_index(i, j)
        self.QD[j],self.QD[i] = self.QD[i],self.QD[j]

class SVR_Q(Kernel):
    #int
    l = None
    #Cache
    cache = None
    #byte[]
    sign = None
    #int[]
    index = None
    #int
    next_buffer = None
    #float[][]
    buffer = None
    #double[]
    QD = None
    
    """
    Parameters:
        svm_problem prob
        svm_parameter param
    """
    def __init__(self, prob, param):
        #super(prob.l, prob.x, param)
        Kernel.__init__(self, prob.l, prob.x, param)
        self.l = prob.l
        self.cache = Cache(self.l, long(param.cache_size * (1 << 20)) )
        self.QD = [0.]*(2*self.l) #new double[2 * l]
        self.sign = [0]*(2*self.l) #new byte[2 * l]
        self.index = [0]*(2*self.l) #new int[2 * l]
        for k in xrange( 0, self.l ): 
            self.sign[k] = 1
            self.sign[k + self.l] = -1
            self.index[k] = k
            self.index[k + self.l] = k
            self.QD[k] = self.kernel_function(k, k)
            self.QD[k + self.l] = self.QD[k]

        self.buffer = [[0.]*(2*self.l), [0.]*(2*self.l)]#new float[2][2 * l]
        self.next_buffer = 0

    def swap_index(self, i, j):
        self.sign[i],self.sign[j] = self.sign[j],self.sign[i]
        self.index[j],self.index[i] = self.index[i],self.index[j]
        self.QD[j],self.QD[i] = self.QD[i],self.QD[j]

    def get_Q(self, i, len):
        """
        Returns float[]
        Parameters:
            i: int  
            len: int
        """
        data = [[0.]*len]#new float[1][]

        real_i = self.index[i]
        if self.cache.get_data(real_i, data, self.l) < self.l: 
            for j in xrange(0, self.l):
                data[0][j] = float(self.kernel_function(real_i, j)) 

        buf = self.buffer[self.next_buffer] #float
        self.next_buffer = 1 - self.next_buffer
        
        si = self.sign[i] #byte
        for j in xrange(0, len):
            buf[j] = float(si)  * self.sign[j] * data[0][self.index[j]]
        return buf

    def get_QD(self):
        return self.QD

class decision_function:
    def __init__(self,alpha,rho):
        self.alpha = alpha
        self.rho = rho
    
    def __repr__(self):
        return "<decision_function(alpha=%s, rho=%s)>" % (self.alpha, self.rho)

def info(s): print s

def solve_c_svc(prob, param, alpha, si, Cp, Cn):
    """
    Parameters:
        prob: svm_problem   
        param: svm_parameter    
        alpha: double[]     
        si: Solver.SolutionInfo     
        Cp: double  
        Cn: double
    """
    l = prob.l
    minus_ones = [0.]*l #new double[l]
    y = [0]*l #new byte[l]

    for i in xrange(0, l): 
        alpha[i] = 0
        minus_ones[i] = -1
        if prob.y[i] > 0: 
            y[i] = +1
        else:
            y[i] = -1

    Solver().solve(l, SVC_Q(prob, param, y), minus_ones, y, alpha, Cp, Cn, param.eps, si, param.shrinking)
    sum_alpha = 0.
    for i in xrange( 0, l ): sum_alpha += alpha[i]
    if Cp == Cn: info("nu = " + str(sum_alpha / (Cp * prob.l)) + "")
    for i in xrange( 0, l ): alpha[i] *= y[i]

def solve_nu_svc(prob, param, alpha, si):
    """
    Parameters:
        prob: svm_problem
        param: svm_parameter
        alpha: double[]
        si: Solver.SolutionInfo
    """
    l = prob.l
    nu = param.nu
    y = [0]*l #new byte[l]
    for i in xrange(0, l): 
        if prob.y[i] > 0: 
            y[i] = +1
        else:
            y[i] = -1
    sum_pos = nu * l / 2.
    sum_neg = nu * l / 2.
    for i in xrange(0, l):
        if y[i] == +1: 
            alpha[i] = min(1.0, sum_pos)
            sum_pos -= alpha[i]
        else:
            alpha[i] = min(1.0, sum_neg)
            sum_neg -= alpha[i]

    zeros = [0.]*l #new double[l]
    Solver_NU().solve(l, SVC_Q(prob, param, y), zeros, y, alpha, 1.0, 1.0, param.eps, si, param.shrinking)
    r = si.r
    info("C = %s " % (1. / r))
    for i in xrange(0, l):
        alpha[i] *= y[i] / r
        si.rho /= r
        si.obj /= (r * r)
        si.upper_bound_p = 1 / r
        si.upper_bound_n = 1 / r

def solve_one_class(prob, param, alpha, si):
    """
    Parameters:
        prob: svm_problem   
        param: svm_parameter   
        alpha: double[]  
        si: Solver.SolutionInfo
    """
    l = prob.l
    zeros = [0.]*l
    ones = [1]*l

    n = int((param.nu * prob.l)) 
    for i in xrange( 0, n ) : 
        alpha[i] = 1
    if n < prob.l: 
        alpha[n] = param.nu * prob.l - n
    for i in xrange( n + 1,  l ) : 
        alpha[i] = 0

    Solver().solve(l, ONE_CLASS_Q(prob, param), zeros, ones, alpha, 1.0, 1.0, param.eps, si, param.shrinking)

def solve_epsilon_svr(prob, param, alpha, si):
    """
    Parameters:
        prob: svm_problem   
        param: svm_parameter    
        alpha: double[] 
        si: Solver.SolutionInfo
    """
    l = prob.l
    alpha2 = [0.]*(2*l) #new double[2 * l]
    linear_term = [0.]*(2*l) #new double[2 * l]
    y = [0]*(2*l) #new byte[2 * l]

    for i in xrange( 0, l ) : 
        alpha2[i] = 0
        linear_term[i] = param.p - prob.y[i]
        y[i] = 1
        alpha2[i + l] = 0
        linear_term[i + l] = param.p + prob.y[i]
        y[i + l] = -1

    Solver().solve(2 * l, SVR_Q(prob, param), linear_term, y, alpha2, param.C, param.C, param.eps, si, param.shrinking)
    sum_alpha = 0.
    for i in xrange( 0, l ) : 
        alpha[i] = alpha2[i] - alpha2[i + l]
        sum_alpha += abs(alpha[i])

    info("nu = %s " % (sum_alpha / (param.C * l)))

def solve_nu_svr(prob, param, alpha, si):
    """
    Parameters:
        prob: svm_problem   
        param: svm_parameter    
        alpha: double[]     
        si: Solver.SolutionInfo
    """
    l = prob.l
    C = param.C
    alpha2 = [0.]*(2*l) #new double[2 * l]
    linear_term = [0.]*(2*l) #new double[2 * l]
    y = [0]*(2*l) #new byte[2 * l]

    sum = C * param.nu * l / 2.
    for i in xrange( 0, l ) : 
        alpha2[i] = alpha2[i + l] = min(sum, C)
        sum -= alpha2[i]
        linear_term[i] = -prob.y[i]
        y[i] = 1
        linear_term[i + l] = prob.y[i]
        y[i + l] = -1

    Solver_NU().solve(2 * l, SVR_Q(prob, param), linear_term, y, alpha2, C, C, param.eps, si, param.shrinking)
    info("epsilon = %s " % (-si.r))
    for i in xrange( 0, l ) : 
        alpha[i] = alpha2[i] - alpha2[i + l]

def svm_train_one(prob, param, Cp, Cn):
    """
    Returns decision_function
    Parameters:
        prob: svm_problem
        param: svm_parameter
        Cp: double
        Cn: double
    """
    alpha = [0.]*prob.l
    si = Solver.SolutionInfo()
    if param.svm_type == SVM_C_SVC:
        solve_c_svc(prob, param, alpha, si, Cp, Cn)
    elif param.svm_type == SVM_NU_SVC:
        solve_nu_svc(prob, param, alpha, si)
    elif param.svm_type == SVM_ONE_CLASS:
        solve_one_class(prob, param, alpha, si)
    elif param.svm_type == SVM_EPSILON_SVR:
        solve_epsilon_svr(prob, param, alpha, si)
    elif param.svm_type == SVM_NU_SVR:
        solve_nu_svr(prob, param, alpha, si)
            
    info("obj = " + str(si.obj) + ", rho = " + str(si.rho) + "")
    nSV = 0
    nBSV = 0
    for i in xrange( 0, prob.l ) : 
        if abs(alpha[i]) > 0: 
            nSV += 1
            if prob.y[i] > 0: 
                if abs(alpha[i]) >= si.upper_bound_p: 
                    nBSV += 1
            else:
                if abs(alpha[i]) >= si.upper_bound_n: 
                    nBSV += 1

    info("nSV = %s, nBSV = %s " % (nSV, nBSV))
    return decision_function(alpha, si.rho)

def sigmoid_train(l, dec_values, labels, probAB):
    """
    Parameters:
        l: int
        dec_values: double[]
        labels: double[]
        probAB: double[]
    """
    A = 0.
    B = 0.

    prior1 = 0.
    prior0 = 0.
    
    max_iter = 100
    min_step = 1.0E-10
    sigma = 1.0E-12
    eps = 1.0E-5

    for i in xrange(0, l):
        if labels[i] > 0: 
            prior1 += 1
        else:
            prior0 += 1
    
    hiTarget = (prior1 + 1.) / (prior1 + 2.)
    loTarget = 1. / (prior0 + 2.)
    t = [0.]*l #new double[l]
    fApB = 0.

    p = None
    q = None
    h11 = None
    h22 = None
    h21 = None
    g1 = None
    g2 = None
    det = None
    dA = None
    dB = None
    gd = None
    stepsize = 0

    # double
    newA = 0.
    newB = 0.
    newf = 0.
    d1 = 0.
    d2 = 0.

    iter = 0

    A = 0.
    B = math.log((prior0 + 1.) / (prior1 + 1.))
    fval = 0.
    for i in xrange(0, l):
        if labels[i] > 0: 
            t[i] = hiTarget
        else:
            t[i] = loTarget
            
        fApB = dec_values[i] * A + B
        if fApB >= 0: 
            fval += t[i] * fApB + math.log(1 + math.exp(-fApB))
        else:
            fval += (t[i] - 1) * fApB + math.log(1 + math.exp(fApB))

    for iter in xrange( 0,  max_iter) : 
        h11 = sigma
        h22 = sigma
        h21 = 0.
        g1 = 0.
        g2 = 0.
        for i in xrange(0, l):
            fApB = dec_values[i] * A + B
            if fApB >= 0.: 
                p = math.exp(-fApB) / (1. + math.exp( -fApB ))
                q = 1. / (1. + math.exp(-fApB))
            else:
                p = 1. / (1. + math.exp(fApB))
                q = math.exp(fApB) / (1. + math.exp( fApB ))

            d2 = p * q
            h11 += dec_values[i] * dec_values[i] * d2
            h22 += d2
            h21 += dec_values[i] * d2
            d1 = t[i] - p
            g1 += dec_values[i] * d1
            g2 += d1

        if abs(g1) < eps and abs(g2) < eps: break
        
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        stepsize = 1
        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = 0.0
            for i in xrange(0, l):
                fApB = dec_values[i] * newA + newB
                if fApB >= 0: 
                    newf += t[i] * fApB + math.log(1 + math.exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + math.log(1 + math.exp(fApB))

            if newf < fval + 1.0E-4 * stepsize * gd: 
                A = newA
                B = newB
                fval = newf
                break
            else:
                stepsize /= 2.

        if stepsize < min_step: 
            info("Line search fails in two-class probability estimates\n")
            break

    if iter >= max_iter: 
        info("Reaching maximal iterations in two-class probability estimates\n")
    probAB[0] = A
    probAB[1] = B

def multiclass_probability(k, r, p):
    """
    Returns void
    Parameters:
        k: intr: double[][]
        p: double[]
    """
    iter = 0
    max_iter = max(100, k)
    Q = a2d(k,k,0.)#new double[k][k]
    Qp = [0.]*k #new double[k]
    pQp = 0.

    eps = 0.0050 / k
    for t in xrange(0, k):
        p[t] = 1.0 / k
        Q[t][t] = 0
        for j in xrange(0, t):
            Q[t][t] += r[j][t] * r[j][t]
            Q[t][j] = Q[j][t]

        for j in xrange(t+1, k):
            Q[t][t] += r[j][t] * r[j][t]
            Q[t][j] = -r[j][t] * r[t][j]

    for iter in xrange(0, max_iter) : 
        pQp = 0
        for t in xrange(0, k):
            Qp[t] = 0
            for j in xrange(0, k):
                Qp[t] += Q[t][j] * p[j]
                pQp += p[t] * Qp[t]

        max_error = 0.
        for t in xrange( 0, k): 
            error = abs(Qp[t] - pQp)
            if error > max_error: 
                max_error = error

        if max_error < eps: 
            break
        for t in xrange( 0, k) : 
            diff = (-Qp[t] + pQp) / Q[t][t]
            p[t] += diff
            pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff)
            for j in xrange( 0, k ) : 
                Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff)
                p[j] /= (1 + diff)

    if iter >= max_iter: 
        info("Exceeds max_iter in multiclass_prob\n")

def svm_binary_svc_probability(prob, param, Cp, Cn, probAB):
    """
    Parameters:
        prob: svm_problem 
        param: svm_parameter 
        Cp: double 
        Cn: double 
        probAB: double[]
    """
    nr_fold = 5
    perm = [0]*prob.l #new int[prob.l]
    dec_values = [0.]*(prob.l) #new double[prob.l]
    
    perm = [i for i in xrange(0, l)]
    for i in xrange(0, prob.l):
        j = i + rand.nextInt(prob.l - i)
        perm[i],perm[j] = perm[j],perm[i]

    for i in xrange(0, nr_fold):
        begin = i * prob.l / nr_fold
        end = (i + 1) * prob.l / nr_fold

        subprob = svm_problem()
        subprob.l = prob.l - (end - begin)
        subprob.x = [[]]*subprob.l #new svm_node[subprob.l][]
        subprob.y = [0.]*subprob.l #new double[subprob.l]
        k = 0
        for j in xrange(0, begin):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        for j in xrange(end, prob.l):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        p_count = 0
        n_count = 0
        for j in xrange(0, k):
            if subprob.y[j] > 0: 
                p_count += 1
            else:
                n_count += 1
        if p_count == 0 and n_count == 0: 
            for j in xrange(begin, end):
                dec_values[perm[j]] = 0
        else:
            if p_count > 0 and n_count == 0: 
                for j in xrange(begin, end):
                    dec_values[perm[j]] = 1
            else:
                if p_count == 0 and n_count > 0: 
                    for j in xrange(begin, end):
                        dec_values[perm[j]] = -1
                else:
                    subparam = svm_parameter(param.clone()) 
                    subparam.probability = 0
                    subparam.C = 1.0
                    subparam.nr_weight = 2
                    subparam.weight_label = [0]*2 #new int[2]
                    subparam.weight = [0.]*2 #new double[2]
                    subparam.weight_label[0] = +1
                    subparam.weight_label[1] = -1
                    subparam.weight[0] = Cp
                    subparam.weight[1] = Cn

                    submodel = svm_train(subprob, subparam)
                    for j in xrange(begin, end):
                        dec_value = [0.] #new double[1]
                        svm_predict_values(submodel, prob.x[perm[j]], dec_value)
                        dec_values[perm[j]] = dec_value[0]
                        dec_values[perm[j]] *= submodel.label[0]

    sigmoid_train(prob.l, dec_values, prob.y, probAB)

def svm_svr_probability(prob, param):
    """
    Returns double
    Parameters:
        prob: svm_problem param: svm_parameter
    """
    nr_fold = 5
    ymv = [0.]*prob.l #new double[prob.l]
    mae = 0.

    newparam = svm_parameter(param.clone()) 
    newparam.probability = 0
    svm_cross_validation(prob, newparam, nr_fold, ymv)
    for i in xrange(0, prob.l):
        ymv[i] = prob.y[i] - ymv[i]
        mae += abs(ymv[i])
    mae /= prob.l
    std = math.sqrt(2 * mae * mae)
    count = 0
    mae = 0.
    for i in xrange(0, prob.l):
        if abs(ymv[i]) > 5 * std: 
            count = count + 1
        else:
            mae += abs(ymv[i])
    mae /= (prob.l - count)
    info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae + "\n")
    return mae

def svm_group_classes(prob, nr_class_ret, label_ret, start_ret, count_ret, perm):
    """
    Parameters:
        prob: svm_problem nr_class_ret: int[] label_ret: int[][] start_ret: int[][] count_ret: int[][] perm: int[]
    """
    l = prob.l
    max_nr_class = 16
    nr_class = 0
    label = [0]*max_nr_class #new int[max_nr_class]
    count = [0]*max_nr_class #new int[max_nr_class]
    data_label = [0]*l #new int[l]

    for i in xrange(0, l):
        this_label = int(prob.y[i]) 

        j = 0
        for jj in xrange(0, nr_class):
            if this_label == label[jj]: 
                count[jj] += 1
                break
            j += 1
                
        data_label[i] = j
        if j == nr_class: 
            if nr_class == max_nr_class: 
                max_nr_class *= 2
                new_data = copy.deepcopy(label)
                #System.arraycopy(label, 0, new_data, 0, label.length)
                label = new_data
                #new_data = new int[max_nr_class]
                #System.arraycopy(count, 0, new_data, 0, count.length)
                count = copy.deepcopy(count)

            label[nr_class] = this_label
            count[nr_class] = 1
            nr_class += 1

    start = [0]*nr_class
    start[0] = 0
    for i in xrange( 1, nr_class ) : 
        start[i] = start[i - 1] + count[i - 1]
    for i in xrange( 0, l ) : 
        perm[start[data_label[i]]] = i
        start[data_label[i]] += 1

    start[0] = 0
    for i in xrange( 1, nr_class ) : 
        start[i] = start[i - 1] + count[i - 1]
    nr_class_ret[0] = nr_class
    label_ret[0] = label
    start_ret[0] = start
    count_ret[0] = count

def svm_train(prob, param):
    """
    Returns svm_model
    Parameters:
        prob: svm_problem param: svm_parameter
    """
    model = svm_model()
    model.param = param
    if param.svm_type in [SVM_ONE_CLASS, SVM_EPSILON_SVR, SVM_NU_SVR]: 
        model.nr_class = 2
        model.label = None
        model.nSV = None
        model.probA = None
        model.probB = None
        model.sv_coef = [[0.]]#new double[1][]
        if param.probability == 1 and (param.svm_type in [SVM_EPSILON_SVR, SVM_NU_SVR]): 
            model.probA = [0.] #new double[1]
            model.probA[0] = svm_svr_probability(prob, param)

        # decision_function
        f = svm_train_one(prob, param, 0, 0)
        model.rho = [f.rho] #new double[1]
        nSV = 0

        for i in xrange(0, prob.l):
            if abs(f.alpha[i]) > 0: 
                nSV += 1
        model.l = nSV
        model.SV = [[]]*nSV #new svm_node[nSV][]
        model.sv_coef[0] = [0.]*nSV #new double[nSV]

        j = 0
        for i in xrange(0, prob.l):
            if abs(f.alpha[i]) > 0: 
                model.SV[j] = prob.x[i]
                model.sv_coef[0][j] = f.alpha[i]
                j += 1

    else:#classification
        l = prob.l
        tmp_nr_class = [0] #new int[1]
        tmp_label = [[]] #new int[1][]
        tmp_start = [[]] #new int[1][]
        tmp_count = [[]] #new int[1][]
        perm = [0]*l #new int[l]
        
        svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm)
        #print tmp_start, tmp_count
        nr_class = tmp_nr_class[0]
        label = tmp_label[0]
        start = tmp_start[0]
        count = tmp_count[0]
        x = [[]]*l #new svm_node[l][]

        for i in xrange(0, l):
            x[i] = prob.x[perm[i]]
            
        #calculate weighted C
        weighted_C = [0.]*nr_class #new double[nr_class]
        for i in xrange(0, nr_class):
            weighted_C[i] = param.C
        for i in xrange(0, param.nr_weight):
            j = 0
            for jj in xrange(0, nr_class):
                if param.weight_label[i] == label[jj]: 
                    break
                j += 1
            if j == nr_class: 
                print "warning: class label " + param.weight_label[i] + " specified in weight is not found\n"
            else:
                weighted_C[j] *= param.weight[i]

        #train k*(k-1)/2 models
        nonzero = [False]*l #new boolean[l]
        
        fn = nr_class * (nr_class - 1) / 2
        f = [None]*fn #new decision_function[nr_class * (nr_class - 1) / 2]
        
        probA = None #double[]
        probB = None #double[]
        if param.probability == 1: 
            np = nr_class * (nr_class - 1) / 2
            probA = [0.]*np
            probB = [0.]*np

        p = 0
        for i in xrange( 0, nr_class ): 
            for j in xrange( i + 1, nr_class ): 
                sub_prob = svm_problem()
                si = start[i]
                sj = start[j]
                ci = count[i]
                cj = count[j]
                sub_prob.l = ci + cj
                sub_prob.x = [[]]*sub_prob.l #new svm_node[sub_prob.l][]
                sub_prob.y = [0.]*sub_prob.l #new double[sub_prob.l]
                #print "svm_train sub_prob", sub_prob, ci, cj, si, sj, sub_prob.l

                for k in xrange(0, ci):
                    sub_prob.x[k] = x[si + k]
                    sub_prob.y[k] = +1

                for k in xrange(0, cj):
                    #print "svm_train", ci+k, sj+k, len(sub_prob.x), len(x)
                    sub_prob.x[ci + k] = x[sj + k]
                    sub_prob.y[ci + k] = -1

                if param.probability == 1: 
                    probAB = [0.]*2 #new double[2]
                    svm_binary_svc_probability(sub_prob, param, weighted_C[i], weighted_C[j], probAB)
                    probA[p] = probAB[0]
                    probB[p] = probAB[1]

                f[p] = svm_train_one(sub_prob, param, weighted_C[i], weighted_C[j])
                #print "f[p]", f[p]
                for k in xrange(0, ci):
                    if not nonzero[si + k] and abs(f[p].alpha[k]) > 0: 
                        nonzero[si + k] = True
                for k in xrange(0, cj):
                    if not nonzero[sj + k] and abs(f[p].alpha[ci + k]) > 0: 
                        nonzero[sj + k] = True
                p += 1

        model.nr_class = nr_class
        model.label = [0]*nr_class #new int[nr_class]
        for i in xrange(0, nr_class):
            model.label[i] = label[i]
        nr = nr_class * (nr_class - 1) / 2
        model.rho = [0.]*nr #new double[nr_class * (nr_class - 1) / 2]
        for i in xrange(0, nr):
            model.rho[i] = f[i].rho
        if param.probability == 1: 
            model.probA = [0.]*nr #new double[nr_class * (nr_class - 1) / 2]
            model.probB = [0.]*nr #new double[nr_class * (nr_class - 1) / 2]
            for i in xrange(0, nr):
                model.probA[i] = probA[i]
                model.probB[i] = probB[i]
        else:
            model.probA = None
            model.probB = None

        nnz = 0
        nz_count = [0]*nr_class #new int[nr_class]
        model.nSV = [0]*nr_class #new int[nr_class]
        for i in xrange(0, nr_class):
            nSV = 0
            for j in xrange(0, count[i]): 
                if nonzero[start[i] + j]: 
                    nSV += 1
                    nnz += 1

            model.nSV[i] = nSV
            nz_count[i] = nSV

        info("Total nSV = " + str(nnz) + "")
        model.l = nnz
        model.SV = [[]]*nnz #new svm_node[nnz][]
        p = 0
        for i in xrange(0, l):
            if nonzero[i]: 
                model.SV[p] = x[i]
                p += 1

        nz_start = [0]*nr_class #new int[nr_class]
        for i in xrange(1, nr_class):
            nz_start[i] = nz_start[i - 1] + nz_count[i - 1]
        model.sv_coef = [[]]*(nr_class-1) #new double[nr_class - 1][]
        for i in xrange(0, nr_class-1):
            model.sv_coef[i] = [0.]*nnz #new double[nnz]
        p = 0
        for i in xrange(0, nr_class):
            for j in xrange(i+1, nr_class):
                si = start[i]
                sj = start[j]
                ci = count[i]
                cj = count[j]
                q = nz_start[i]

                for k in xrange(0, ci):
                    if nonzero[si + k]: 
                        model.sv_coef[j - 1][q+1] = f[p].alpha[k]
                        q += 1
                q = nz_start[j]
                for k in xrange(0, cj):
                    if nonzero[sj + k]: 
                        model.sv_coef[i][q] = f[p].alpha[ci + k]
                        q += 1
                p += 1
    return model

def svm_cross_validation(prob, param, nr_fold, target):
    """
    Parameters:
        prob: svm_problem   
        param: svm_parameter   
        nr_fold: int    
        target: double[]
    """
    fold_start = [0]*(nr_fold+1) #new int[nr_fold + 1]
    l = prob.l
    perm = [0]*l
    if (param.svm_type in [SVM_C_SVC, SVM_NU_SVC]) and nr_fold < l: 
        tmp_nr_class = [0] #new int[1]
        tmp_label = [[0]] #new int[1][]
        tmp_start = [[0]] #new int[1][]
        tmp_count = [[0]] #new int[1][]
        svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm)
        nr_class = tmp_nr_class[0]
        start = tmp_start[0]
        count = tmp_count[0]
        fold_count = [0]*nr_fold #new int[nr_fold]

        index = [0]*l #new int[l]
        for i in xrange(0, l):
            index[i] = perm[i]
        for c in xrange(0, nr_class):
            for i in xrange(0, count[c]):
                j = i + rand.nextInt(count[c] - i)
                index[start[c] + i],index[start[c] + j] = index[start[c] + j],index[start[c] + i]
                
        for i in xrange(0, nr_fold):
            fold_count[i] = 0
            for c in xrange(0, nr_class):
                fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold

        fold_start[0] = 0
        for i in xrange(1, nr_fold+1):
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1]
        for c in xrange(0, nr_class):
            for i in xrange(0, nr_fold):
                begin = start[c] + i * count[c] / nr_fold
                end = start[c] + (i + 1) * count[c] / nr_fold
                for j in xrange(begin, end):
                    perm[fold_start[i]] = index[j]
                    fold_start[i] += 1

        fold_start[0] = 0
        for i in xrange(1, nr_fold+1):
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1]
    else:
        for i in xrange(0, l):
            perm[i] = i
        for i in xrange(0, l):
            j = i + rand.randint(l - i)
            perm[j],perm[i] = perm[i], perm[j]

        for i in xrange(0, nr_fold+1):
            fold_start[i] = i * l / nr_fold

    for i in xrange(0, nr_fold):
        begin = fold_start[i]
        end = fold_start[i + 1]

        subprob = svm_problem()
        subprob.l = l - (end - begin)
        subprob.x = [[]]*subprob.l #new svm_node[subprob.l][]
        subprob.y = [0.]*subprob.l #new double[subprob.l]
        k = 0
        for j in xrange(0, begin):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        for j in xrange(end, l):
            subprob.x[k] = prob.x[perm[j]]
            subprob.y[k] = prob.y[perm[j]]
            k += 1

        submodel = svm_train(subprob, param)
        if param.probability == 1 and (param.svm_type in [SVM_C_SVC, SVM_NU_SVC]): 
            prob_estimates = [0.]*submodel.nr_class 
            for j in xrange(begin, end):
                target[perm[j]] = svm_predict_probability(submodel, prob.x[perm[j]], prob_estimates)
        else:
            for j in xrange(begin, end):
                target[perm[j]] = svm_predict(submodel, prob.x[perm[j]])

def svm_predict_values(model, x, dec_values):
    """
    Returns double
    Parameters:
        model: svm_model
        x: svm_node[]
        dec_values: double[]
    """
    if model.param.svm_type in [SVM_ONE_CLASS, SVM_EPSILON_SVR, SVM_NU_SVR]: 
        # double[]
        sv_coef = model.sv_coef[0]
        # double
        sum = 0.
        for i in xrange(0, model.l):
            sum += sv_coef[i] * Kernel.k_function(x, model.SV[i], model.param)
        sum -= model.rho[0]
        dec_values[0] = sum
        if model.param.svm_type == SVM_ONE_CLASS: 
            if sum > 0:
                return 1
            else:
                return -1
        else:
            return sum
    else:
        nr_class = model.nr_class
        l = model.l

        kvalue = [0.]*l #new double[l]
        for i in xrange(0, l):
            kvalue[i] = Kernel.k_function(x, model.SV[i], model.param)

        start = [0]*nr_class #new int[nr_class]
        for i in xrange(1, nr_class):
            start[i] = start[i - 1] + model.nSV[i - 1]

        vote = [0]*nr_class #new int[nr_class]
        for i in xrange(0, nr_class):
            vote[i] = 0
            
        p = 0
        for i in xrange(0, nr_class):
            for j in xrange(i+1, nr_class):
                sum = 0.
                si = start[i]
                sj = start[j]
                ci = model.nSV[i]
                cj = model.nSV[j]

                # double[]
                coef1 = model.sv_coef[j - 1]
                coef2 = model.sv_coef[i]
                for k in xrange(0, ci):
                    sum += coef1[si + k] * kvalue[si + k]
                for k in xrange(0, cj):
                    sum += coef2[sj + k] * kvalue[sj + k]
                sum -= model.rho[p]
                dec_values[p] = sum
                if sum > 0: 
                    vote[i] += 1
                else:
                    vote[j] += 1
                p += 1

        vote_max_idx = 0
        for i in xrange(1, nr_class):
            if vote[i] > vote[vote_max_idx]: 
                vote_max_idx = i
        return model.label[vote_max_idx]

def svm_predict(model, x):
    """
    Returns double
    Parameters:
        model: svm_model 
        x: svm_node[]
    """
    
    if model.get_type() in [SVM_ONE_CLASS, SVM_EPSILON_SVR, SVM_NU_SVR]: 
        dec_values = [0.] #new double[1]
    else:
        nv = model.nr_class*(model.nr_class-1)/2
        dec_values = [0.]*nv # new double[nr_class * (nr_class - 1) / 2]
    
    return svm_predict_values(model, x, dec_values)

def svm_predict_probability(model, x, prob_estimates):
    """
    Returns double
    Parameters:
        model: svm_modelx: svm_node[]prob_estimates: double[]
    """
    def sigmoid_predict(decision_value, A, B):
        fApB = decision_value * A + B
        if fApB >= 0.: 
            return math.exp(-fApB) / (1. + math.exp( -fApB ))
        else:
            return 1. / (1 + math.exp( fApB ))
    
    if (model.param.svm_type in [SVM_C_SVC, SVM_NU_SVC]) and model.probA is not None and model.probB is not None: 
        nr_class = model.nr_class
        # double[]
        dec_values = [0.]*nr_class * (nr_class - 1) / 2 #new double[nr_class * (nr_class - 1) / 2]
        svm_predict_values(model, x, dec_values)
        min_prob = 1.0E-7
        pairwise_prob = a2d(nr_class,nr_class,0.) #new double[nr_class][nr_class]
        # int
        k = 0
        for i in xrange(0, nr_class):
            for j in xrange(i+1, nr_class):
                pairwise_prob[i][j] = min( max( sigmoid_predict(dec_values[k], model.probA[k], model.probB[k]), min_prob), 1.-min_prob)
                pairwise_prob[j][i] = 1. - pairwise_prob[i][j]
                k += 1

        multiclass_probability(nr_class, pairwise_prob, prob_estimates)
        # int
        prob_max_idx = 0
        for i in xrange(1, nr_class):
            if prob_estimates[i] > prob_estimates[prob_max_idx]: 
                prob_max_idx = i
        return model.label[prob_max_idx]
    else:
        return svm_predict(model, x)

def svm_check_parameter(prob, param):
    """
    Returns String
    Parameters:
        prob: svm_problem 
        param: svm_parameter
    """
    svm_type = param.svm_type
    if svm_type not in [SVM_C_SVC, SVM_NU_SVC, SVM_ONE_CLASS, SVM_EPSILON_SVR, SVM_NU_SVR]: 
        return "unknown svm type"
    if param.kernel_type not in [SVM_LINEAR, SVM_POLY, SVM_RBF, SVM_SIGMOID, SVM_PRECOMPUTED]: 
        return "unknown kernel type"
    if param.gamma < 0: 
        return "gamma < 0"
    if param.degree < 0: 
        return "degree of polynomial kernel < 0"
    if param.cache_size <= 0: 
        return "cache_size <= 0"
    if param.eps <= 0: 
        return "eps <= 0"
    if svm_type in [SVM_C_SVC, SVM_EPSILON_SVR, SVM_NU_SVR]: 
        if param.C <= 0: return "C <= 0"
    if svm_type in [SVM_NU_SVC, SVM_ONE_CLASS, SVM_NU_SVR]: 
        if param.nu <= 0 or param.nu > 1: 
            return "nu <= 0 or nu > 1"
    if svm_type == SVM_EPSILON_SVR: 
        if param.p < 0: return "p < 0"
    if param.shrinking != 0 and param.shrinking != 1: 
        return "shrinking != 0 and shrinking != 1"
    if param.probability != 0 and param.probability != 1: 
        return "probability != 0 and probability != 1"
    if param.probability == 1 and svm_type == SVM_ONE_CLASS: 
        return "one-class SVM probability output not supported yet"
    if svm_type == SVM_NU_SVC: 
        l = prob.l
        max_nr_class = 16
        nr_class = 0
        label = [0]*max_nr_class
        count = [0]*max_nr_class
        for i in xrange(0, l): 
            this_label = int(prob.y[i]) 

            for j in xrange(0, nr_class): 
                if this_label == label[j]: 
                    count[j] += 1
                    break

                if j == nr_class: 
                    if nr_class == max_nr_class: 
                        max_nr_class *= 2
                        # int[]
                        #new_data = new int[max_nr_class]
                        #System.arraycopy(label, 0, new_data, 0, label.length)
                        label = copy.deepcopy(label)
                        #new_data = new int[max_nr_class]
                        #System.arraycopy(count, 0, new_data, 0, count.length)
                        count = copy.deepcopy(count)

                    label[nr_class] = this_label
                    count[nr_class] = 1
                    nr_class += 1

        for i in xrange(0, nr_class): 
            n1 = count[i]
            for j in xrange(i + 1, nr_class): 
                n2 = count[j]
                if param.nu * (n1 + n2) / 2 > min(n1, n2): 
                    return "specified nu is infeasible"

    return None