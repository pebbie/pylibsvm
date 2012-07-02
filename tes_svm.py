from svm import *

def load_problem(filename):
    prob = svm_problem()
    f = open(filename)
    prob.y = []
    prob.x = []
    for line in f.readlines():
        tokens = line.strip().split(" ")
        prob.y.append(int(tokens[0]))
        tmp = []
        for ti in xrange(1, len(tokens)):
            pair = tokens[ti].split(":")
            tmp.append(svm_node(index=int(pair[0]), value=float(pair[1])))
        prob.x.append(tmp)
    f.close()
    prob.l = len(prob.y)
    return prob

def xor_problem():
    problem = svm_problem()
    problem.y = [
                -1, 
                1, 
                1, 
                -1]
    problem.x = [
                [svm_node(index=0, value=0), svm_node(index=1, value=0)],
                [svm_node(index=0, value=0), svm_node(index=1, value=1)],
                [svm_node(index=0, value=1), svm_node(index=1, value=0)],
                [svm_node(index=0, value=1), svm_node(index=1, value=1)]
                ]
    problem.l = len(problem.y)
    return problem

problem = load_problem('iris.scale')
param = svm_parameter()
param.kernel_type = SVM_POLY
param.gamma = 1./len(problem.x[0])
param.degree = 1
model = svm_train(problem, param)

success = 0
for i in xrange(0, problem.l):
    if svm_predict(model, problem.x[i])==problem.y[i] : success += 1
    #print svm_predict(model, problem.x[i]), problem.y[i], svm_predict(model, problem.x[i])==problem.y[i]
print "success rate : %s %%" % (success*100./problem.l)
model.save('iris.model')
model.load('iris.model')
model.save('iris2.model')
