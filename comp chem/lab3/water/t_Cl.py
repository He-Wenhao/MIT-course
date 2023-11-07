import csv
import matplotlib.pyplot as plt
import numpy as np


R_lst = []
O_lst = []
File = open("water_rdf_OCl")
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


## first peak: 3.142
## first minimum: 3.812 ind:152
## second peak: 4.889
## second minimum: 6.187 ind:257