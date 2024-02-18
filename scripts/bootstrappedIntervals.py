import numpy, scipy.stats, math

bootstrap = 100000
lag = 7
N = 10000

if N*bootstrap*8 > 1e10:
	print 'Too high N'
	exit()

percs = [0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.99]

Ks = [scipy.stats.norm.ppf(perc)*math.sqrt(lag) for perc in percs]

inds = numpy.array([[x for x in range(i,N-(lag-i-1))] for i in range(0,lag)]).transpose()
random = numpy.random.normal(size=N*bootstrap)
ones = numpy.ones(lag)
returns = numpy.array([numpy.dot(random[N*i:N*(i+1)][inds],ones) for i in range(0,bootstrap)])
observations = [numpy.sum(returns<K,1) for K in Ks]

intervals = [(numpy.percentile(observ,2.5), numpy.percentile(observ,97.5)) for observ in observations]
