#!/bin/env python2
'''
A script simulating how the modulator works in a 2-D GC
'''
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate



peak_range = 3.

class GC1D:
    '''
    The 1D GC distribution class.
    '''
    lowb = 0
    highb = 0
    def __init__(self, retensions, sigmas, partition, lowb=0, highb=0):
        self.retensions = retensions
        self.sigmas = sigmas
        self.partition = partition
        self.xfunc = lambda x:gaussian_sum(x, retensions, sigmas, partition)
        self.func = np.vectorize(self.xfunc)
        self.lowb = lowb
        self.highb = highb
        self.duration = highb - lowb

    def signal(self, time):
        '''
        Just a wrapper of the signal function
        '''
        return self.func(time)

    def set_duration(self, dur):
        self.duration = dur
        return self.duration

    def make_division(self, lowb=0, highb=0, interval=0,
                    method='Traditional'):
        '''
        make divisions of two methods
        the returning value is just the data type of corresponding methods
        '''
        if (self.lowb != 0 and lowb == 0):
            lowb = self.lowb
        if self.highb != 0 and highb == 0:
            highb = self.highb

        if method == 'Traditional':
            assert interval != 0
            return self.make_divide_trad(lowb, highb, interval)
        elif method == 'Smart':
            return self.make_divide_smart(lowb, highb)


    def make_divide_trad(self, lowb, highb, interval=5):
        '''
        make divisions for Traditional method
        '''
        return np.arange(lowb, highb, interval, dtype=np.float)

    def make_divide_smart(self, lowb, highb):
        '''
        make divisions for Smart method
        '''
        # peak_range controls the width of each peak
        # the interval of each peak is tR \pm peak_range \times \sigma
        heads1 = np.subtract(self.retensions, np.multiply(self.sigmas, peak_range))
        tails1 = np.add(self.retensions, np.multiply(self.sigmas, peak_range))
        divisions = zip(heads1, tails1)
        return merge(divisions)

    def divide(self, chop_time, retensions2, sigmas2, method='Traditional'):
        '''
        This function simulates the modulator.
        Given the method (Traditional or Smart) and an array of chop_time,
        it will return an array of GC1D functions
        '''
        assert(len(retensions2) == len(self.retensions))
        assert(len(retensions2) == len(sigmas2))
        if method == 'Traditional':
            # which means every point is a division point of the modulator,
            # the second column will accept sample flow all the time
            next_col = []
            for i in range(len(chop_time)-1):
                part = []
                for j in range(len(self.retensions)):
                    quant_pass, Equant_pass = integrate.quad(
                            gaussian, chop_time[i], chop_time[i+1],
                            args=(self.retensions[j], self.sigmas[j])
                        )
                    quant_pass = quant_pass * self.partition[j]
                    part.append(quant_pass)

                next_col.append(GC1D(retensions2, sigmas2, part,
                                     lowb=chop_time[i], highb=chop_time[i+1]))
            return next_col

        elif method == 'Smart':
            # which means the chop_time array should appear in pairs
            # (namely a 2xN matrix)
            next_col = []
            for i in range(len(chop_time)):
                part = []
                for j in range(len(self.retensions)):
                    quant_pass, Equant_pass = integrate.quad(
                        gaussian, chop_time[i][0], chop_time[i][1],
                        args=(self.retensions[j], self.sigmas[j])
                        )
                    quant_pass = quant_pass * self.partition[j]
                    part.append(quant_pass)
                tmp1d = GC1D(retensions2, sigmas2, part,
                                     lowb=chop_time[i][0], highb=chop_time[i][1])
                next_col.append(tmp1d)
            return next_col

    def make_graph(self, second_col, Nsample2=100, method="Traditional", duration=0, Nsample1=500):
        '''
        This function calculates the real signal on a 2-D mesh grid
        '''
#         lowbs = [obj.lowb for obj in second_col]
#         highbs = [obj.highb for obj in second_col]
        retensions2 = second_col[0].retensions
        sigmas2 = second_col[0].sigmas
        tails2 = np.add(retensions2, np.multiply(sigmas2, 3. ))
        duration = np.max(tails2)
        # data = []
        if method == 'Traditional':
            return self.make_graph_trad(second_col, Nsample2, duration)
        elif method == 'Smart':
            return self.make_graph_smart(second_col,
                                         duration, Nsample2, Nsample1)
        else:
            return ()

    def make_graph_trad(self, second_col, Nsample2=100, duration=0):
        lowbs = [obj.lowb for obj in second_col]
        highbs = [obj.highb for obj in second_col]
        time1 = np.divide(np.add(lowbs, highbs), 2.)
        time2 = np.linspace(0, duration, Nsample2)
        data = []
        for i in range(len(second_col)):
            data = np.concatenate([data, second_col[i].func(time2)])
        data = np.reshape(data, (len(second_col), Nsample2))
        return (time1, time2, np.transpose(data))

    def make_graph_smart(self, second_col, duration, Nsample2=100, Nsample1=500):
        lowbs = [obj.lowb for obj in second_col]
        highbs = [obj.highb for obj in second_col]
        tsample1 = np.linspace(self.lowb, self.highb, Nsample1)

#         original_data = self.func(tsample1)
        data = np.zeros(Nsample1 * Nsample2)
        time2 = np.linspace(0, duration, Nsample2)
        for i in range(len(second_col)):
            # calculate conditional indices: element being true means this sample point is within
            # the i th sampling range of the modulator
            # length of selected_ind is Nsample1
            selected_ind = (tsample1>=lowbs[i]) & (tsample1<highbs[i])
            # selected_data = original_data[selected_ind]
#             quant_pass, Equant_pass = integrate.quad(
#                 self.func, lowbs[i], highbs[i])
#             scalar = np.divide(original_data, quant_pass)
            col2_data = second_col[i].func(time2)
#             fig = plt.figure()
#             plt.plot(time2, col2_data)
#             fig.savefig('curve_smart_%00d.eps')
            selected_ind = np.tile(selected_ind, Nsample2)
            gauss_mu = (lowbs[i] + highbs[i]) / 2
            gauss_sigma = (highbs[i] - lowbs[i]) / peak_range / 2.
            vgaussian = np.vectorize(lambda x: gaussian(x, gauss_mu, gauss_sigma))
            scalar = vgaussian(tsample1)
            scalar = np.tile(scalar, Nsample2)
            col2_data = np.repeat(col2_data, Nsample1)
            ############ for debugging
            # print np.shape(scalar), np.shape(col2_data), np.shape(selected_ind), np.shape(data)
            ############
            data = np.where(selected_ind, np.multiply(scalar, col2_data), data)

        data = np.reshape(data, (Nsample2, Nsample1))
        return (tsample1, time2, data)


def gaussian(time, mu, sigma):
    '''
    The gaussian distribution function
    '''
    return np.exp( -np.power((time-mu)/sigma, 2) / 2) / np.sqrt(2*np.pi) / sigma

def gaussian_sum(time, retensions, sigmas, partition):
    assert len(retensions) == len(sigmas)
    assert len(retensions) == len(partition)
    intens = 0
    for i in range(len(retensions)):
        compo = partition[i] * np.exp(-((time-retensions[i]) / sigmas[i])**2 / 2
                                  ) / np.sqrt(2*np.pi) / sigmas[i]
        intens = intens + compo

    return intens

def plot_surf(Ax, Ay, surf, **kwargs):
    '''
    Plot a surface with matplotlib,
    accepts 2 required args,
    Ax, Ay are arrays of points along each axis;
    surf is a matrix of corresponding dimensions.
    It also accepts several keyword args that controls plotting,
    for more detailed description of the keyword args, refer to matplotlib documentations
    '''
    Mx, My = np.meshgrid(Ax, Ay)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if 'figname' in kwargs:
        figname = kwargs.pop('figname')
    else:
        figname = 'gc2d.png'

    # one set of recommended arguments is as follows
    # ax.plot_surface(Mx, My, surf,
    #     cmap=matplotlib.cm.jet, rstride=2, cstride=10, linewidth=0, kwargs)
    ax.plot_surface(Mx, My, surf, **kwargs)

    ax.set_xlabel('t1')
    ax.set_ylabel('t2')
    ax.set_zlabel('Intensity')
    plt.show()
    fig.savefig(figname)

def write_data(Ax, Ay, data, filename='graph.dat'):
    datafile = open(filename, 'w')
    Mx, My = np.meshgrid(Ax, Ay)
    assert np.shape(Mx) == np.shape(data)
    for i in range(len(Mx)):
        Xwrite, Ywrite, Zwrite = Mx[i], My[i], data[i]
        for j in range(len(Xwrite)):
            datafile.write('%g\t%g\t%g\n' % (Xwrite[j], Ywrite[j], Zwrite[j]))

        datafile.write('\n')
    datafile.close()

def merge(times):
    sorted_list = sorted([sorted(t) for t in times])
    result = []
    i = 0
    while (i < len(sorted_list)):
        if i > len(sorted_list):
            break
        interv = sorted_list[i]
        while(i<len(sorted_list) and sorted_list[i][0] < interv[1] ):
            interv[1] = max(interv[1], sorted_list[i][1])
            i = i+1
        result.append(tuple(interv))
    return result


def calcRetension( gas_vis, col_len, Pst, flow_rate,
              delta_S, delta_H, delta_Cp, col_rad, coating_thick, T, R):
    T0 = 298.15
    A = (delta_S - delta_Cp * np.log(T0) - delta_Cp)/R
    B = (delta_H - delta_Cp * T0)/R
    C = delta_Cp / R
    K = np.exp(A - B/T + C * np.log(T))
    beta = (col_rad - coating_thick)**2/(2*col_rad*coating_thick)
    tm = 8/3 * np.sqrt(np.pi*col_len^3*gas_vis*T0/(Pst*flow_rate*T))
    return tm * (1 + K/beta)

def calcSigma( tR, delta_S, delta_H, delta_Cp, col_rad,
              coating_thick, plate_num,
              mod_period=0, method="Traditional"):
    T0 = 298.15
    R = 8.314
    T = T0
    A = (delta_S - delta_Cp * np.log(T) - delta_Cp)/R
    B = (delta_H - delta_Cp * T0)/R
    C = delta_Cp / R
    K = np.exp(A - B/T + C * np.log(T))
    beta = (col_rad - coating_thick)**2/(2*col_rad*coating_thick)
    if method == 'Smart':
        return 4 * tR * (1+K/beta) / np.sqrt(plate_num)
    else:
        tmpv = tR * (1+K/beta) / np.sqrt(plate_num)
        return 4 * np.sqrt(tmpv ** 2 + (mod_period/2)**2)

