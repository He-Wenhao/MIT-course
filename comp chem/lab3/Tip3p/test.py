# %%
import csv
import matplotlib.pyplot as pllt

# %%
File = open("summary.DENSITY.csv")
Reader = csv.reader(File)
Reader = list(Reader)


# %%
density_lst = []
for i in Reader[1:]:
    density_lst.append(float(i[0]))
print(len(density_lst))


