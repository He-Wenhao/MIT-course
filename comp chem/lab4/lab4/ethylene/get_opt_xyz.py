# read output file
with open('ethylene_optimize.out','r') as f:
    line = next(f)
    # find OPTIMIZATION CONVERGED
    while 'OPTIMIZATION CONVERGED' not in line:
        line = next(f)
    # skip ahead to xyz coordinates
    for _ in range(6):
        line = next(f)
    # save coordinates in list, removing numbering
    xyz = []
    while line:
        temp = line.split()
        if not temp:
            break
        elif not temp[0].isnumeric():
            break
        natoms = temp[0]
        # save line as string, with new line character
        xyz.append(' '.join(temp[1:])+'\n')
        line = next(f)

# write xyz file
with open('opt_ethylene.xyz','w') as f:
    f.write(f'{natoms}\n\n')
    f.writelines(xyz)
# write QCHEM molecule file
with open('ethylene_molecule','w') as f:
    f.write('$molecule\n   0 1\n')
    for line in xyz:
        f.write('   '+line)
    f.write('$end')

