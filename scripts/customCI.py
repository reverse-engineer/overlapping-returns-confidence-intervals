import argparse, numpy, re, scipy.stats, itertools

parser = argparse.ArgumentParser(description=r'''Script to compute analytical confidence intervals for aggregated overlapping or non-overlapping back-tests.
Please do not forget to provide in the same folder, a folder "pdfs", with the analytical pdfs for overlapping returns with all the lags you are requesting.''')

parser.add_argument('codes', metavar='NO[lag]', type=str, nargs='+', help='For each backtest to aggregate specify a string: NO[lag], where N is the number of observations in the back-test, O is the overlapping indicator (Y or N), and lag is the lag of the overlapping returns. For example, 55N 242Y7 aggregates two backtests: one with 55 non-overlapping observations, and one with 242 7-day overlapping observations')
args = parser.parse_args()

args.percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def convolute(pdf1,pdf2):
	Ntot = pdf1.shape[0] + pdf2.shape[0] - 1
	Nold = pdf1.shape[0]
	Nnew = pdf2.shape[0]
	return numpy.sum(numpy.array([[(pdf1[z-y,:] if z-y >= 0 and z-y<Nold else 0)*pdf2[y,:] for z in range(0, Ntot)] for y in range(0, Nnew)]),0)

Pdfs = []

for code in args.codes:
	overl = code.find('Y')
	limit = max(overl, code.find('N'))
	if limit <= 0:
		raise ValueError('code :' + str(code))
	Pdfs.append(numpy.load('pdfs/' + code[limit+1:len(code)] + '/pdf' + code[0:limit] +'.bin') if overl > -1 else numpy.array([scipy.stats.binom.pmf(i,int(code[0:limit]),perc) for perc, i in itertools.product(args.percentiles, range(0,int(code[0:limit])+1))]).reshape((len(args.percentiles), int(code[0:limit])+1)).transpose())

cumpdf = Pdfs.pop()

while len(Pdfs) > 0:
	cumpdf = convolute(cumpdf, Pdfs.pop())

cumcumpdf = numpy.cumsum(cumpdf,0)
leftlow = numpy.where(cumcumpdf<=0.025)
rightlow = numpy.where(cumcumpdf>=0.975)
print '\n'.join([str(i) + ' ' + str(j) for i, j in zip([max(leftlow[0][leftlow[1]==i] if leftlow[0][leftlow[1]==i].shape[0]>0 else [0]) for i in range(0,12)],[min(rightlow[0][rightlow[1]==i] if rightlow[0][rightlow[1]==i].shape[0]>0 else [0]) for i in range(0,12)])])