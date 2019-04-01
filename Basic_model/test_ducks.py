import numpy as np
import yaml

def read_duck():
	print("[ data ] Now reading the Duck dataset...")
	f=open("./ducks/labels.txt")
	ducks_info=f.readlines()
	workers_num=40
	tasks_num=240
	edges_type=2

	f2=open("./ducks/map.yaml")
	gtLabels = yaml.load(f2)
	workerId=gtLabels['wkr']
	wkr_num=len(workerId.keys())
	workerId2Idx=dict((id, idx) for (idx, id) in enumerate(workerId.values()))
	print(workerId2Idx)
	graph=np.zeros(wkr_num*240)
	graph.shape=[wkr_num,240]
	graph-=1
	#count_wkr=np.zeros(60)
	print["[ data ] Building the Graph......"]
	for i in range(len(ducks_info)):
		x = ducks_info[i]
		#print(x)
		x = str(x).split("\n")[0]#.split[" "]]
		x = [int(j) for j in str(x).split(" ")]
		#count_wkr[x[1]]+=1

		graph[workerId2Idx[x[1]]-1][x[0]-1]=x[2]
		#print(x)
	print("[ data ] Graph built finished.")
	f3=open("./ducks/classes.yaml")
	Labels = yaml.load(f3)
	true_labels = np.zeros(240)
	for i in range(240):
		true_labels[i]=Labels[i]

	print("[ data ]  Now getting the true labels.")

	return graph.shape, wkr_num,240,2,graph,true_labels