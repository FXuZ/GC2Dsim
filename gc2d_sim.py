#!/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
from gc2d_utils import *

data_filename = "./valid_data.csv"
# geometric parameters of the 2D GC
col_len1 = 30
col_dia1 = 0.32e-3
col_len2 = 1.5
col_dia2 = 0.1e-3
col_N1 = np.sqrt(col_len1 / col_dia1)
col_N2 = np.sqrt(col_len2 / col_dia2)

def gc2d_sim():
    # read data from given file
    datafile = open(data_filename, 'r')
    gas_partition = []
    gases = []
    retensions1 = []
    retensions2 = []
    for line in datafile:
        gas_prop = line.strip().split(';')
        gases.append(gas_prop[0])
        retensions1.append(float(gas_prop[1]))
        retensions2.append(float(gas_prop[2]))
    # calculate necessary parameters
    sigmas1 = np.multiply( retensions1, 2.75)/ col_N1
    sigmas2 = np.multiply( retensions2, 2.75) / col_N2
    tails1 = np.multiply(sigmas1, 3) + retensions1
    heads1 = retensions1 - np.multiply( sigmas1, 3)
    if len(gas_partition) == 0:
        gas_partition = np.random.rand(len(retensions1))

    # construct the data of the first detector
    lowb = min(heads1)
    highb = max(tails1)
    detect1 = GC1D(retensions1, sigmas1, gas_partition, lowb, highb)
    detect2 = detect1.divide(
        detect1.make_division(lowb, highb, interval=6, method='Traditional'),
        retensions2, sigmas2, method="Traditional")
    detect2S = detect1.divide(
        detect1.make_division(lowb, highb, precision=1.e-3, method='Smart'),
        retensions2, sigmas2, method="Smart")
    (time1, time2, data) = detect1.make_graph(detect2S, 100, method="Smart")
    plot_surf(time1, time2, data, cmap=matplotlib.cm.jet, rstride=2, cstride=10, linewidth=0)
    write_data(time1, time2, data)
    # write 1st dimensional data to a text file
    d1data = detect1.signal(time1)
    d1filename = 'D1signal.dat'
    d1datafile = open(d1filename, 'w')
    d1datafile.write('time\tsignal')
    for i in range(len(time1)):
        d1datafile.write('%g\t%g' % (time1[i], d1data[i]))
    d1datafile.close()
    # print np.shape(time1)
    # print np.shape(time2)
    # print np.shape(data)
    # print time1, time2, data


if __name__ == '__main__':
    gc2d_sim()
