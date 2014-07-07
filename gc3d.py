#!/bin/env python2

import gc2d_utils
import numpy as np
from scipy import integrate

class GC3D:
    def init(self, retensions1, sigmas1, retensions2, sigmas2, retensions3, sigmas3, partition, lowb, highb):
        self.retensions1 = retensions1
        self.retensions2 = retensions2
        self.retensions3 = retensions3
        self.sigmas1 = sigmas1
        self.sigmas2 = sigmas2
        self.sigmas3 = sigmas3
        self.partition = partition
        self.lowb = lowb
        self.highb = highb
        self.detect1 = gc2d_utils.GC1D(retensions1, sigmas1, partition, lowb, highb)
        self.detect2 = self.detect1.divide(
            self.detect1.make_division(lowb, highb, method='Smart'),
            retensions2, sigmas2, method='Smart')
        self.detect3 = [ pk.divide(
                pk.make_division(pk.lowb, pk.highb, method='Smart'),
                retensions3, sigmas3, method='Smart')
                for pk in self.detect2]
        self.lowb1 = self.detect1.lowb
        self.highb1 = self.detect1.highb
        self.lowbs2 = [obj.lowb for obj in self.detect2]
        self.highbs2 = [obj.highb for obj in self.detect2]
        self.lowb2 = 0 # min( [obj.lowb for obj in self.detect2] )
        self.highb2 = np.amax( self.highbs2 )
        self.lowbs3 = ([[pk.lowb for pk in dim2] for dim2 in self.detect3])
        self.highbs3 = ([[pk.highb for pk in dim2] for dim2 in self.detect3])
        self.lowb3 = 0 # np.amin(self.lowbs3)
        self.highb3 = np.amax(self.highbs3)
        def signal(t1, t2, t3):
            for i in range( len(self.detect3) ):
                if t1 <= self.highbs2[i] and t1 >= self.lowbs2[i]:
                    gauss_mu1 = ( self.highbs2[i] + self.lowbs2[i] ) / 2
                    gauss_sigma1 = ( self.highbs2[i] - self.lowbs2[i] ) / gc2d_utils.peak_range / 2
                    for j in range( len(self.detect3[i]) ):
                        if t2 <= self.highbs3[i][j] and t2 >= self.lowbs3[i][j]:
                            gauss_mu2 = ( self.highbs3[i][j] + self.lowbs3[i][j] ) / 2
                            gauss_sigma2 = ( self.highbs3[i][j] - self.lowbs3[i][j] ) / gc2d_utils.peak_range / 2
                            sig3 = self.detect3[i][j].func(t3) * gc2d_utils.gaussian(
                                    t1, gauss_mu1, gauss_sigma1) * gc2d_utils.gaussian(t2, gauss_mu2, gauss_sigma2)
                            return sig3
            return 0

        self.signal3D = signal

    def make_data(self, Nsample1, Nsample2, Nsample3):
        '''
        make_data: take 3 integers as arguments as the number of sample points along each dimension,
                    returning a 4-element tuple consisting of the sampling time of dimension and
                    the data taken by the simulated process of a smart GC

                This function simulates the process of taking data. Please note that the first and the second
                dimensions' data are reformed and approximated by single gaussian peaks which are right
                at the center of the intervals chopped by the modulators. If this is not wanted, changing the
                definition of the scale1 and scale2 variables.
        '''
        data = np.zeros( Nsample1 * Nsample2 * Nsample3 )
        time1 = np.linspace( self.lowb1, self.highb1, Nsample1 )
        time2 = np.linspace( self.lowb2, self.highb2, Nsample2 )
        time3 = np.linspace( self.lowb3, self.highb3, Nsample3 )
        for i in range(len(self.detect3)):
            ind_mask1 = ( time1 >= self.lowbs2[i] ) & ( time1 <= self.highbs2[i] )
            gauss_mu1 = ( self.lowbs2[i] + self.highbs2[i] ) / 2
            gauss_sigma1 = ( self.highbs2[i] - self.lowbs2[i] ) / gc2d_utils.peak_range / 2
            vgaussian1 = np.vectorize( lambda x: gc2d_utils.gaussian(x, gauss_mu1, gauss_sigma1 ) )
            scale1 = vgaussian1( time1 )
            scale1 = np.repeat( Nsample2 * Nsample3 )
            ind_mask1 = np.repeat( Nsample2 * Nsample3 )
            for j in range(len(self.detect3[i])):
                ind_mask2 = ( time2 >= self.lowbs3[i][j] ) & ( time2 <= self.highbs3[i][j])
                gauss_mu2 = ( self.lowbs3[i][j] + self.highbs2[i][j] ) / 2
                gauss_sigma2 = ( self.highbs2[i][j] - self.lowbs2[i][j] ) / gc2d_utils.peak_range / 2
                vgaussian2 = np.vectorize( lambda x: gc2d_utils.gaussian(x, gauss_mu2, gauss_sigma2 ) )
                scale2 = vgaussian2( time2 )
                scale2 = np.tile( np.repeat( scale2, Nsample3 ), Nsample2 )
                ind_mask2 = np.tile( np.repeat( ind_mask2, Nsample3 ), Nsample2 )
                ind_mask = np.logical_and( ind_mask1, ind_mask2 )
                col3_curve = np.tile( self.detect3[i][j]( time3 ), Nsample1*Nsample2 )
                data = np.where(ind_mask, np.multiply( np.multiply( scale1, scale2 ), col3_curve ), data )

        data = np.reshape(data, Nsample1, Nsample2, Nsample3)
        return (time1, time2, time3, data)

    def projectionX(self, Nsample1, Nsample2, Nsample3 ):
        time2 = np.linspace(self.lowb2, self.highb2, Nsample2)
        time3 = np.linspace(self.lowb3, self.highb3, Nsample3)
        time23X, time23Y = np.meshgrid(time2, time3)
        projection = np.zeros(np.size(time23X))
        for i in range(len(time23X)):
            for j in range(len(time23X[i])):
                curve1 = lambda x: self.signal3D(x, time23X[i][j], time23Y[i][j])
                projection[i][j] = integrate.quad( curve1, self.lowb1, self.highb)
        return projection

    def projectionY(self, Nsample1, Nsample2, Nsample3 ):
        time1 = np.linspace(self.lowb1, self.highb1, Nsample1)
        time3 = np.linspace(self.lowb3, self.highb3, Nsample3)
        time13X, time13Y = np.meshgrid(time1, time3)
        projection = np.zeros(np.size(time13X))
        for i in range(len(time13X)):
            for j in range(len(time13X[i])):
                curve2 = lambda x: self.signal3D(x, time13X[i][j], time13Y[i][j])
                projection[i][j] = integrate.quad( curve2, self.lowb1, self.highb)
        return projection

    def projectionZ(self, Nsample1, Nsample2, Nsample3 ):
        time1 = np.linspace(self.lowb1, self.highb1, Nsample1)
        time2 = np.linspace(self.lowb2, self.highb2, Nsample2)
        time12X, time12Y = np.meshgrid(time1, time2)
        projection = np.zeros(np.size(time12X))
        for i in range(len(time12X)):
            for j in range(len(time12X[i])):
                curve3 = lambda x: self.signal2D(x, time12X[i][j], time12Y[i][j])
                projection[i][j] = integrate.quad( curve3, self.lowb1, self.highb)
        return projection

    def sliceX(self, axis, time, Nsample1, Nsample2, Nsample3 ):
        time2 = np.linspace(self.lowb2, self.highb2, Nsample2)
        time3 = np.linspace(self.lowb3, self.highb3, Nsample3)
        time23X, time23Y = np.meshgrid(time2, time3)
        return (np.vectorize(self.signal3D)) (time, time23X, time23Y)

    def sliceY(self, axis, time, Nsample1, Nsample2, Nsample3 ):
        time2 = np.linspace(self.lowb2, self.highb2, Nsample2)
        time3 = np.linspace(self.lowb3, self.highb3, Nsample3)
        time23X, time23Y = np.meshgrid(time2, time3)
        return (np.vectorize(self.signal3D)) (time, time23X, time23Y)

    def sliceZ(self, axis, time, Nsample1, Nsample2, Nsample3 ):
        time2 = np.linspace(self.lowb2, self.highb2, Nsample2)
        time3 = np.linspace(self.lowb3, self.highb3, Nsample3)
        time23X, time23Y = np.meshgrid(time2, time3)
        return (np.vectorize(self.signal3D)) (time, time23X, time23Y)

def sim_test():
    data_filename = "./data.csv"
    # geometric parameters of the 3D GC
    col_len1 = 30
    col_dia1 = 0.32e-3
    col_len2 = 1.5
    col_dia2 = 0.1e-3
    col_len3 = 1.5
    col_dia3 = 0.1e-3
    col_N1 = np.sqrt(col_len1 / col_dia1)
    col_N2 = np.sqrt(col_len2 / col_dia2)
    col_N3 = np.sqrt(col_len3 / col_dia3)
    datafile = open(data_filename, "r")
    gas_partition = []
    gases = []
    retensions1 = []
    retensions2 = []
    retensions3 = []
    for line in datafile:
        gas_prop = line.strip().split(';')
        gases.append(gas_prop[0])
        retensions1.append(float(gas_prop[1]))
        retensions2.append(float(gas_prop[2]))
        retensions3.append(float(gas_prop[3]))
    # calculate necessary parameters
    sigmas1 = np.multiply( retensions1, 2.75) / col_N1
    sigmas2 = np.multiply( retensions2, 2.75) / col_N2
    sigmas3 = np.multiply( retensions3, 2.75) / col_N3
    tails1 = np.multiply(sigmas1, 3) + retensions1
    heads1 = retensions1 - np.multiply( sigmas1, 3)
    lowb = min(heads1)
    highb = max(tails1)
    if len(gas_partition) == 0:
        gas_partition = np.random.rand(len(retensions1))

    GC_test = GC3D(retensions1, sigmas1, retensions2, sigmas2, retensions3, sigmas3, gas_partition, lowb, highb)

    (time1, time2, time3, data) = GC_test.make_data(100, 100, 100)
    output_filename = "3DGC_simul.dat"
    output_file = open(output_filename, 'w')
    for i in range(len(time1)):
        for j in range(len(time2)):
            for k in range(len(time3)):
                output_file.write("%g\t%g\t%g\t%g\n", time1[i], time2[j], time3[k], data[i][j][k])

if __name__ == "__main__":
    sim_test()
