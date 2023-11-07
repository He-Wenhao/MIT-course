import csv
import matplotlib.pyplot as plt
import numpy as np


R_lst = []
O_lst = []
File = open("water_rdf_OMG")
Reader = csv.reader(File)
Reader = list(Reader)
for i in Reader[1:]:
    i_lst = i[0].strip().split()
    R_lst.append(float(i_lst[0]))
    O_lst.append(float(i_lst[1]))
plt.plot(R_lst,O_lst)
plt.title('O-MG- radial distribution function')
plt.xlabel('r(Ang)')
plt.ylabel('g(r)')
plt.show()


## first peak: 2.038
## first minimum: 2.800 ind:111
## second peak: 4.214
## second minimum: 5.143 ind:205