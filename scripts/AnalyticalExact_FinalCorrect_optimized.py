
# -> not really well optimized... TODO! -> the loop on line 50 is the bottleneck and can be improved.
import scipy.stats, numpy, math, itertools

def convolute(pdf1,pdf2):
	Ntot = pdf1.shape[0] + pdf2.shape[0] - 1
	Nold = pdf1.shape[0]
	Nnew = pdf2.shape[0]
	return numpy.sum(numpy.array([[(pdf1[z-y,:] if z-y >= 0 and z-y<Nold else 0)*pdf2[y,:] for z in range(0, Ntot)] for y in range(0, Nnew)]),0)

def unique_rows(a):
	a = numpy.ascontiguousarray(a)
	unique_a, inverse_index = numpy.unique(a.view([('', a.dtype)]*a.shape[1]),return_inverse=True)
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1])), inverse_index

def joint_distribution(lag,autocorrellag,ks):
	if lag > 15:
		print "Lag too big, this will take ages... max 15!"
		exit()
	covar = numpy.zeros((lag,lag,lag))
	for i in range(1,autocorrellag+1):
		if autocorrellag-i < lag:
			covar[:,:,i+(lag-autocorrellag)-1] = numpy.diag(numpy.ones(i+(lag-autocorrellag))*i,k=autocorrellag-i)
	covar = numpy.sum(covar+covar.transpose((1,0,2)),2)-numpy.diag(numpy.diag(covar[:,:,lag-1]))
	joint = []
	#precalculate the heavy stuff
	allc = [comb for comb in itertools.product([0,1],repeat=lag)]
	for K in ks:
		joint.append([scipy.stats.mvn.mvnun([-50 if c else K for c in comb],[K if c else 50 for c in comb],numpy.zeros(lag),covar,abseps=1e-10,releps=1e-10)[0] for comb in allc])
	#temporary fix: divide by sum to ensure sum to 1 of pdf... -> check why this is not the case! Probably numerical precision of mvnun!!!
	return numpy.array(joint).transpose()/sum(numpy.array(joint).transpose()), numpy.array(allc)

def sum_dist_from_joint(joint, combs):
	sumcombs = numpy.sum(combs,1)
	vals = numpy.unique(sumcombs)
	return vals, numpy.array([numpy.sum(joint[sumcombs==val,:],0) for val in vals])
	
def compute_pdfs(Lag, percs, Ntot):
	Ks = [scipy.stats.norm.ppf(perc)*math.sqrt(Lag) for perc in percs]
	FinalN = []
	Joints = []
	
	jointCumul, combsCumul = joint_distribution(Lag+1,Lag,Ks)
	jointSingleNew, combsSingleNew = joint_distribution(Lag,Lag,Ks)
	jointCommon, combsCommon = joint_distribution(Lag-1,Lag,Ks)
	Joints.append((jointCumul, combsCumul))
	FinalN.append((Lag+1, sum_dist_from_joint(jointCumul,combsCumul)))
	for N in range(0,Ntot):
		combsSumFirstTwo = numpy.concatenate((numpy.sum(combsCumul[:,0:2],1).reshape((combsCumul.shape[0],1)),combsCumul[:,2:combsCumul.shape[1]]),1)
		combsSumFirstTwoUnique, indices = unique_rows(combsSumFirstTwo)
		jointSumFirstTwo = numpy.array([numpy.sum(jointCumul[indices==i,:],0) for i in range(0,max(indices)+1)])
		jointCumul = numpy.concatenate((numpy.array([((jointSFT).transpose()*jointSingleNew[numpy.all(combsSingleNew == numpy.concatenate((combSFT[1:combSFT.shape[0]],[1])),1),:] \
			/ jointCommon[numpy.all(combsCommon == combSFT[1:combSFT.shape[0]],1),:] \
			).transpose() \
			for combSFT, jointSFT in zip(combsSumFirstTwoUnique,jointSumFirstTwo)]), \
			numpy.array([((jointSFT).transpose()*jointSingleNew[numpy.all(combsSingleNew == numpy.concatenate((combSFT[1:combSFT.shape[0]],[0])),1),:] \
			/ jointCommon[numpy.all(combsCommon == combSFT[1:combSFT.shape[0]],1),:] \
			).transpose() \
			for combSFT, jointSFT in zip(combsSumFirstTwoUnique,jointSumFirstTwo)])),0)
		#Normalize because of numerical instability
		jointCumul = jointCumul/sum(jointCumul)
		combsCumul = numpy.concatenate((numpy.concatenate((combsSumFirstTwoUnique,numpy.ones((combsSumFirstTwoUnique.shape[0],1),dtype='int32')),1), \
			numpy.concatenate((combsSumFirstTwoUnique,numpy.zeros((combsSumFirstTwoUnique.shape[0],1),dtype='int32')),1)),0)
		Joints.append((jointCumul, combsCumul))
		FinalN.append((N+Lag+2, sum_dist_from_joint(jointCumul,combsCumul)))
	return FinalN

percentiles = [0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.99]
# PdfsN = compute_pdfs(7, percentiles, 1000)
# 
# for pdf in PdfsN:
# 	left = numpy.where(numpy.cumsum(pdf[1][1],0)<=0.025)
# 	right = numpy.where(numpy.cumsum(pdf[1][1],0)>=0.975)
# 	print '\n'.join([str(pdf[0]) + ' ' + str(i) + ' ' + str(j) for i, j in zip([max(left[0][left[1]==i] if left[0][left[1]==i].shape[0]>0 else [0]) for i in range(0,12)],[min(right[0][right[1]==i] if right[0][right[1]==i].shape[0]>0 else [0]) for i in range(0,12)])])
# 	numpy.save(open('pdfs/' + str(7) + '/pdf' + str(pdf[0]) + '.bin', 'wb'), pdf[1][1])