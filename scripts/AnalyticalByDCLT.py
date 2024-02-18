import scipy.stats, numpy, math, itertools


lag = 7

percs = [0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.99]

Ks = [scipy.stats.norm.ppf(perc)*math.sqrt(lag) for perc in percs]

covar = numpy.ndarray((lag,lag,lag))
for i in range(1,lag+1):
    covar[:,:,i-1] = numpy.diag(numpy.ones(i)*i,k=lag-i)

covar = numpy.sum(covar+covar.transpose((1,0,2)),2)-numpy.diag(numpy.diag(covar[:,:,lag-1]))

joint = []

combs = []
#precalculate the heavy stuff
allc = [comb for comb in itertools.product([0,1],repeat=lag)]

for K in Ks:
	joint.append([scipy.stats.mvn.mvnun([-50 if c else K for c in comb],[K if c else 50 for c in comb],numpy.zeros(lag),covar)[0] for comb in allc])

joint = numpy.array(joint).transpose()

sumCovars = [sum([sum(joint[numpy.logical_and(numpy.array(allc)[:,0] == 1, numpy.array(allc)[:,i] == 1),:])[K]-percs[K]*percs[K] for i in range(1,lag)]) for K in range(0,len(percs))]

intervals = [[(scipy.stats.norm.ppf(0.025,perc*N,math.sqrt((2*sumCov + (perc-perc*perc))*N)), \
	scipy.stats.norm.ppf(0.975,perc*N,math.sqrt((2*sumCov + (perc-perc*perc))*N))) for perc, sumCov in zip(percs,sumCovars)] for N in range(1,10000)]