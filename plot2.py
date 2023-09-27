import matplotlib.pyplot as plt
import numpy as np  

'''
nq = 10

for nl in [2,4,6]:
	y = np.loadtxt('./002/model_nq_{}/loss_nq_{}_nl_{}_lr_0.001.txt'.format(nq,nq,nl))

	y = np.array(y)
	dy = y.mean(0)
	

	x = np.array([i for i in range(1, len(dy)+1 )])
	plt.plot(x,dy,label='L={}'.format(nl))

	plt.fill_between(x, y.min(0), y.max(0), alpha=0.2)
plt.xticks([1,10,20])


plt.legend()
plt.show()
'''

minimo = 1000
maximo = 0

for nl in [2,4,6]:
	for nq in [2,4,6,8,10]:
		y = np.loadtxt('./008/model_nq_{}/loss_nq_{}_nl_{}_lr_0.001.txt'.format(nq,nq,nl))

		if np.min(y.min(0)) < minimo:
			minimo = np.min(y.min(0))

		if np.max(y.max(0)) > maximo:
			maximo = np.max(y.max(0))


for nl in [2,4,6]:
	for nq in [2,4,6,8,10]:
		y = np.loadtxt('./009/model_nq_{}/loss_nq_{}_nl_{}_lr_0.001.txt'.format(nq,nq,nl))

		if np.min(y.min(0)) < minimo:
			minimo = np.min(y.min(0))

		if np.max(y.max(0)) > maximo:
			maximo = np.max(y.max(0))



for nl in [2,4,6]:

	for nq in [2,4,6,8,10]:
		y = np.loadtxt('./008/model_nq_{}/loss_nq_{}_nl_{}_lr_0.001.txt'.format(nq,nq,nl))
		z = np.loadtxt('./009/model_nq_{}/loss_nq_{}_nl_{}_lr_0.001.txt'.format(nq,nq,nl))

		y = np.array(y)
		dy = y.mean(0)
		dz = z.mean(0)
		

		x = np.array([i for i in range(1, len(dy)+1 )])
		plt.plot(x,dy,label='NQ={} L={} Free'.format(nq,nl))
		plt.fill_between(x, y.min(0), y.max(0), alpha=0.2)
		
		plt.plot(x,dz,label='NQ={} L={} Not Free'.format(nq,nl))
		plt.fill_between(x, z.min(0), z.max(0), alpha=0.2)
		

		plt.xticks([1,10,20])

		if nq == 10:
			plt.xlabel('Epochs',fontsize=16)
		if nl == 2:
			plt.ylabel('Loss',fontsize=16)
		plt.ylim(minimo-0.01,maximo+0.001)
		plt.legend()
		plt.savefig('./figuras/nq_{}_L_{}.pdf'.format(nq,nl))
		plt.close()
		#plt.show()

