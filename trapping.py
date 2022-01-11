#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:49:21 2021

@author: jeremy

This file is an attempt to make a well organized class for calculating, displaying and 
playing with dipole trap potentials calculated from comsol simultaions

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nanotrappy as nt
import nanotrappy.utils.physicalunits as pu
import nanotrappy.utils.viz as viz
from nanotrappy.utils import vdw
import warnings
import scipy.optimize as so
from scipy.signal import argrelextrema
import matplotlib.patches as mpl_patches
import ffd_mode_solver_2 as ffd 




class datastruct:
    
    def __init__(self, filename=None, aoi=None, gridsize=200):

        self.aoi = aoi
        self.gridsize = gridsize
        self.convert = np.vectorize(self.convert_s)
        self.filename = filename
        self.wavelength = self.get_wavelength()
        self.width, self.height = self.get_waveguide_dimensions()
        # if aoi is None:
            # self.aoi = [-self.width/2, self.height/2+150e-9, self.width/2, self.height/2]
        self.efield_b1 = self.get_efield(1)
        self.efield_b2 = self.get_efield(1)  # TODO beware, this is temporary and should be changed to 2
        # self.from_csv = False
        # self.df = 0

    def print_(self):
        print(self.filename)
        # print(os.path.join(os.path.dirname(self.filename), )
        h = os.path.basename(self.filename).split("_")
        newstring = "H_"+h[1]+"_W_"+h[3]+"_P.csv"
        print(newstring)

    def get_efield_from_csv(self, eigenfrequency):
        df = pd.read_csv(self.filename, header=8)
        # self.df = df
        data = np.array([self.convert(el) for el in df.to_numpy().transpose()])
        ex = data[3+(eigenfrequency-1)*5]
        ey = data[4+(eigenfrequency-1)*5]
        ez = data[5+(eigenfrequency-1)*5]
        # power = np.average(np.array(data[3+(eigenfrequency-1)*5]))
        directory, filen = os.path.split(self.filename)
        filen = filen.split("_")
        newstring = "H_"+filen[1]+"_W_"+filen[3]+"_P.csv"
        power_data = pd.read_csv(os.path.join(directory, newstring), header=4).to_numpy()
        power = 0
        for line in power_data:
            try:
                np.testing.assert_approx_equal(line[0], self.wavelength, significant=3)
                power = line[-1]
            except AssertionError:
                pass
        x = np.unique(data[0])
        y = np.unique(data[1])
        z = np.array([0])
        E_field = np.zeros((3,len(x),len(y),len(z)),dtype = "complex")
        iterable = 0
        for j in range(len(y)):
            for i in range(len(x)):
                for k in range(len(z)):
                    E_field[:,i,j,k] = ex[iterable], ey[iterable], ez[iterable]
                    iterable+=1
        formated_array = np.array([self.wavelength, x,y,z, E_field])
        return formated_array, power
    
    def get_efield(self, eigenfrequency):
        npyname = self.filename.replace('.csv',"_eigen"+str(eigenfrequency)+'.npy')
        if not os.path.isfile(npyname) and os.path.isfile(self.filename):
            print("Getting data from *.csv")
            efield, power = self.get_efield_from_csv(eigenfrequency)
            efield[4] = efield[4]/np.sqrt(power)
            self.from_csv = True
        elif os.path.isfile(npyname):
            efield = np.load(npyname, allow_pickle=True)
            self.from_csv = True
        else:
            efield = self.calculate_efield_resonator(eigenfrequency)
            self.from_csv = False
        return efield
    
    def calculate_efield_resonator(self, eigenfrequency):
        simu = ffd.simulation(wavelength=self.wavelength, 
                       height=self.height, 
                       width=self.width,
                       gridsize=self.gridsize)
        efield = simu.formated_data[eigenfrequency-1]
        return efield 
    
    def get_subpicture(self, farr):
        print(self.filename)
        aoi =self.aoi 
        x = farr[1] 
        y = farr[2]
        z = farr[3]
        e = farr[4]
        ex = e[0]
        ey = e[1]
        ez = e[2] 
        if aoi is not None:
            xtruth = (x>=aoi[0]) * (x<=aoi[2])
            ytruth = (y<=aoi[1]) * (y>=aoi[3])
        else:
            xtruth = (x==x)
            ytruth = (y==y)
        xnew = x[xtruth]
        ynew = y[ytruth] 
        temp = xtruth 
        xtruth = ytruth
        ytruth = temp
        exnew = np.array([line[ytruth] for line in ex[xtruth]]).transpose(1,0,2)
        eynew = np.array([line[ytruth] for line in ey[xtruth]]).transpose(1,0,2)
        eznew = np.array([line[ytruth] for line in ez[xtruth]]).transpose(1,0,2)
        return np.array([farr[0], xnew, ynew, z, np.array([exnew, eynew, eznew])], dtype=object) 
    
    def export_to_eigenfolders(self, redorblue=''):
        if not self.from_csv:
            f1 = self.get_subpicture(self.efield_b1)
            f2 = self.get_subpicture(self.efield_b2)
        else:
            f1 = self.efield_b1
            f2 = self.efield_b2
        options = ['r1b1','r1b2','r2b1','r2b2']
        for i, o in enumerate(options):
            folder = os.path.join(os.path.dirname(self.filename),"eigen_"+o)
            fileN = os.path.basename(self.filename.replace('.csv',"_eigen"))
            os.makedirs(folder, exist_ok=True)
            fname1 = os.path.join(folder, fileN+"1.npy")
            fname2 = os.path.join(folder, fileN+"2.npy")
            if redorblue=='red':
                if o[1]=='1' and not os.path.isfile(fname1):
                    np.save(fname1, f1)
                elif o[1]=='2' and not os.path.isfile(fname2):
                    np.save(fname2, f2)
            elif redorblue=='blue':
                if o[3]=='1' and not os.path.isfile(fname1):
                    np.save(fname1, f1)
                elif o[3]=='2' and not os.path.isfile(fname2):
                    np.save(fname2, f2)
        
    def convert_s(self, s):
        cn = complex(str(s).replace("i","j"))
        # print(cn)
        return cn
    
    def get_wavelength(self):
        n = os.path.split(self.filename)[1].split('_')[4].split('.csv')[0]
        return float(n)*1e-9
        
    def get_waveguide_dimensions(self):
        n = os.path.split(self.filename)[1].split('H_')[1].split('_W_')
        height = n[0]
        width = n[1].split('_')[0]
        return float(width)*1e-9, float(height)*1e-9
    
    def plot_intensity(self, eigenfrequency, plot=True, cmap='hsv_r'):
        dat = self.efield_b1 if eigenfrequency==1 else self.efield_b2
        x = dat[1]
        y = dat[2]
        z = dat[3]
        e_field = dat[4]
        intensity = np.zeros((len(x), len(y)))



        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    vect = e_field[:,i,j,k]
                    intensity[i,j] = 0.5*pu.cc*np.sqrt(1)* np.linalg.norm(vect)**2 #  TODO: insert effective refractive index here
        if plot:
            plt.imshow(intensity.T,
                       origin='lower', 
                       extent=[x.min()*1e9, x.max()*1e9, y.min()*1e9, y.max()*1e9], 
                       cmap=cmap
                       )
            plt.xlabel("x (nm)")
            plt.ylabel("y (nm)")
            plt.title("Intensity of eigenmode %i at $\\lambda=%i$ nm\nHeight=%i nm, Width=%i nm"
                      %(eigenfrequency, self.wavelength*1e9, self.height*1e9, self.width*1e9))
            plt.colorbar()
            plt.tight_layout()
        return intensity


class trap:
    def __init__(self, 
                 datafolder_red, 
                 datafolder_blue,
                 height = 150e-9,
                 width = 1000e-9,
                 wavelengths = [690e-9, 900e-9],
                 eigenfrequency_r=1,
                 eigenfrequency_b =1,
                 aoi = None,
                 gridsize = 200,
                 ):
        self.gridsize = gridsize
        filename_blue = os.path.join(datafolder_blue, "H_%i_W_%i_%i.csv"%(height*1e9, width*1e9, wavelengths[0]*1e9))
        filename_red = os.path.join(datafolder_red, "H_%i_W_%i_%i.csv"%(height*1e9, width*1e9, wavelengths[1]*1e9))
        self.eigenfrequency_r = eigenfrequency_r
        self.eigenfrequency_b = eigenfrequency_b
        self.red = datastruct(filename_red, aoi, gridsize)
        self.blue = datastruct(filename_blue, aoi, gridsize)
        self.red.export_to_eigenfolders("red")
        self.blue.export_to_eigenfolders("blue")
        self.red_beam = nt.trapping.beam.BeamPair(self.red.wavelength, 1 , self.red.wavelength, 1 )
        self.blue_beam = nt.trapping.beam.Beam(self.blue.wavelength, "f", 1 )
        # self.blue_beam2 = nt.trapping.beam.Beam(self.blue.wavelength+30e9, "b", 1 )
        self.trap = nt.trapping.trap.Trap_beams(self.blue_beam, self.red_beam)
        self.syst = nt.trapping.atomicsystem.atomicsystem(nt.Rubidium87(), "5S1/2", pu.f2)
        self.surface = vdw.PlaneSurface(normal_axis=nt.trapping.geometry.AxisY(), 
                                        normal_coord=self.red.height/2.0)
        self.Simul = nt.trapping.simulation.Simulation(
            self.syst,
            nt.SiN(),
            self.trap,
            os.path.join(os.path.dirname(filename_red),'eigen_r%ib%i'%(eigenfrequency_r,eigenfrequency_b)),
            self.surface,
            )
        self.Simul.geometry = nt.trapping.geometry.PlaneXY(normal_coord=0)
        self.Simul.compute()
        self.Simul.save()
        # self.red_beam.set_power(1)
        self.Simul.total_potential()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.potentials = self.Simul.potentials.astype(float)

    def potential2d(self, power_b=1e-3, power_r=1e-3, mF = 0, remove_center=False, includeVdW=True):
        pot = power_b/2*self.potentials[0,:,:,mF+2]+power_r*self.potentials[1,:,:,mF+2]#+power_b/2*self.potentials[2,:,:,mF+2]
        if includeVdW:
            pot+=self.Simul.CP
        if remove_center:
            for i in range(len(pot)):
                for j in range(len(pot[0])):
                    x = self.Simul.x
                    y = self.Simul.y
                    width = self.red.width
                    height = self.blue.height
                    if abs(x[i])<(width/2) and abs(y[j])<(height/2):
                        pot[i,j]=0
        return pot
    
    def potential1d(self, power_b=1, power_r=1, n_line=-1, includeVdW=True):
        n_line = int(len(self.Simul.y)/2) if n_line==-1 else n_line
        truth = np.where(self.Simul.x>self.red.width/2)
        potentials1d = []
        for i in range(-2,2+1):
            potential = power_b*self.potentials[0,:,:,i+2]+power_r*self.potentials[1,:,:,i+2]
            potentials1d.append(potential[:,n_line][truth])
        if includeVdW:
            potentials1d = [el+self.Simul.CP[:,n_line][truth] for el in potentials1d]
        return self.Simul.x[truth], potentials1d
    
    def potential1dv(self, power_b=1, power_r=1, n_col=-1, includeVdW=True):
        n_col = int(len(self.Simul.x)/2) if n_col==-1 else n_col
        truth = np.where(self.Simul.y>self.red.height/2)
        potentials1d = []
        for i in range(-2,2+1):
            potential = power_b*self.potentials[0,:,:,i+2]+power_r*self.potentials[1,:,:,i+2]
            potentials1d.append(potential[n_col,:][truth])
        if includeVdW:
            potentials1d = [el+self.Simul.CP[n_col,:][truth] for el in potentials1d]
        return self.Simul.y[truth], potentials1d

    def get_potential(self, power_b=1, power_r=1, axis='Y', coordinates=(0,0), includeVdW=True):
        if len(axis)==2 and len(coordinates)==2:
            coordinates=0
        if axis=='Y':
            self.Simul.geometry = nt.trapping.geometry.AxisY(coordinates)
        elif axis=="X":
            self.Simul.geometry = nt.trapping.geometry.AxisX(coordinates)
        elif axis =='Z':
            self.Simul.geometry = nt.trapping.geometry.AxisZ(coordinates)
        elif axis=='XY':
            self.Simul.geometry = nt.trapping.geometry.PlaneXY(coordinates)
        elif axis=='XZ':
            self.Simul.geometry = nt.trapping.geometry.PlaneXZ(coordinates)
        elif axis=='YZ':
            self.Simul.geometry = nt.trapping.geometry.PlaneYZ(coordinates)
        self.red_beam.set_power(power_r)
        self.blue_beam.set_power(power_b)
        self.Simul.compute()
        pot = self.Simul.total_potential()
        if includeVdW:
            return pot[0]+self.Simul.CP
        
        return pot[0]
    
    def get_trap_frequency(self, power_b=1, power_r=1, n_line=-1, plot=False):
        n_line = int(len(self.Simul.y)/2) if n_line==-1 else n_line
        t, pots = self.potential1d(power_b, power_r, n_line)
        if np.argmin(pots[0])==0 or np.argmin(pots[0])==len(t)-1:
            print('No trap minima exists')
            return -1e30
        tmin = t[np.argmin(pots[0])]
        pots = [pot[12:40] for pot in pots]
        # t = t[12:40]-3.25e-7
        t = t[12:40] - tmin
        res =  np.polyfit(t, pots[0], deg=9)
        p = np.poly1d(res)

        # p2 = np.poly1d(res[-2:])
        if plot:
            plt.clf()
            plt.plot(t, pots[0],'.')
            plt.plot(t, p(t))
            plt.plot(t, res[-1]+res[-3]*t**2+res[-2]*t)
        return res[-3]
    
    def find_optimal_power(self, pb=2.5, pr0=1, h_or_v='v'):
        # bnds = (0.0, 10.0)
        func = lambda pr: self.get_trap_depht(pb, pr, for_mimization=True)
        # func = np.vectorize(func)
        res = so.minimize(func, pr0)# , method='Nelder-Mead')  #, bounds = bnds)
        print(res)
        print("Result: %.3f"%res.x[0])
        return res
        # pr = np.arange(0.46,1.35,0.01)
        # plt.plot(pr, func(pr))
        # tf = [self.get_trap_depht(2.5,prr) for prr in pr]
        # plt.plot(pr, tf, '.')
    
    def plot_potential2d(self, power_b=1, power_r=1, mF = 0, remove_center=True, cmap='jet_r', includeVdW=True):
        potential = self.potential2d(power_b, power_r, mF, remove_center, includeVdW=includeVdW)
        if remove_center:
            depth = -self.get_trap_dephtv(power_b, power_r)
            potential[abs(potential)>depth*1.02]=0
            potential[potential>0.001]=0
        x = self.Simul.x
        y = self.Simul.y
        plt.clf()
        plt.imshow(potential.T, 
                   origin='lower', 
                   extent=[x.min(),x.max(),y.min(),y.max()], 
                   cmap=cmap
                   )
        plt.title("Temperature (mK)\nEigenmode %i at $\\lambda=%i$ nm, P=%.1f mW,\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                  %(self.eigenfrequency_b, self.blue.wavelength*1e9, power_b*1e3, 
                    self.eigenfrequency_r, self.red.wavelength*1e9, power_r*1e3))
        # potential[int(len(self.Simul.y)/2)]=np.zeros(len(potential[0]))
        # plt.axvline(y=self.Simul.y[int(len(self.Simul.y)/2)], color='red')
        plt.xlabel("$x$ (m)")
        plt.ylabel("$y$ (m)")

        # img = ax.imshow(potential, extent=[Simul.x.min(),Simul.x.max(),Simul.y.min(),Simul.y.max()])
        plt.colorbar()
        plt.tight_layout()
        
    def plot_potential1d(self, power_b=1, power_r=1):
        plt.clf()
        t, potentials = self.potential1d(power_b, power_r)
        for i, pot in enumerate(potentials):
            plt.plot(t*1e9, pot, label="$m_f=%i$"%(i-2))
        plt.axvline(x=self.red.width/2.0*1e9, color='black')
        plt.legend()
        plt.title("Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                  %(self.eigenfrequency_b, self.blue.wavelength*1e9, power_b, self.eigenfrequency_r, self.red.wavelength*1e9, power_r))
        plt.xlabel("Distance from center of waveguide (nm)")
        plt.ylabel("Potential (a.u.)")
        plt.tight_layout()
        
        
    def plot_potential1dv(self, power_b=1, power_r=1, includeVdW=True):
        plt.clf()
        t, potentials = self.potential1dv(power_b, power_r,includeVdW=includeVdW)
        # for i, pot in enumerate(potentials):
        pot = potentials[2]
        plt.plot((t-self.red.height/2)*1e9, pot)# , label="$m_f=%i$"%(i-2))
        plt.axvline(x=0, color='black')
        # plt.legend()
        plt.title("Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                  %(self.eigenfrequency_b, self.blue.wavelength*1e9,power_b*1e3, self.eigenfrequency_r, self.red.wavelength*1e9,power_r*1e3))
        plt.xlabel("Distance from surface (nm)")
        plt.ylabel("Potential (mK)")
        ax = plt.gca()
        depth = self.get_trap_dephtv(power_b, power_r)
        if depth!=0:
            t, potentials = self.potential1dv(power_b, power_r, includeVdW=includeVdW)
            t = t[:80]
            potentials = [el[:80] for el in potentials]
            # vmax = argrelextrema(potentials[0], np.greater)[0]
            vmin = argrelextrema(potentials[0], np.less)[0]
            position = t[vmin]
            plt.plot((position-self.red.height/2)*1e9, potentials[0][vmin], 'xr')
            # plt.xlabel("Distance from surface (nm)\n\n\nTrap depth %.0f uK\nTrap position %.0f nm"%(-depth*1e3, (position-self.red.height/2)*1e9))#, xy, args, kwargs)
            ax.set_ylim([potentials[0][vmin]*1.05, 50e-3])
            ax.set_xlim([0, 500])
            # create a list with two empty handles (or more if needed)
            handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
            # create the corresponding number of labels (= the text you want to display)
            labels = []
            labels.append("Trap depth %.2f mK"%(-depth))
            labels.append("Trap position %.0f nm"%((np.real(position)-self.red.height/2)*1e9))
            # create the legend, supressing the blank space of the empty line symbol and the
            # padding between symbol and label by setting handlelenght and handletextpad
            ax.legend(handles, labels, loc='best', fontsize='medium', 
                      fancybox=True, framealpha=0.7, 
                      handlelength=0, handletextpad=0)
        plt.tight_layout()
        
        
        
    def get_trap_depht(self, power_b=1, power_r=1, axis='Y', coordinates=(0,0)):
        if len(axis)==2 and len(coordinates)==2:
            coordinates=0        
        pot = self.get_potential(power_b, power_r, axis, coordinates)
        truthiness = self.Simul.y>self.red.width
        y_outside = self.Simul.y[truthiness]
        trap_outside = pot[truthiness]
        vizu = viz.Viz(self.Simul, trapping_axis=axis)
        # res = vizu.get_min_trap(y_outside, trap_outside)
        return [vizu.get_min_trap(y_outside, trap_outside[:,i]) for i in range(len(pot[0]))]
        
    def get_trap_dephtv(self, power_b, power_r, includeVdW=True, for_minimization=False):
        t, potentials = self.potential1dv(power_b, power_r, includeVdW=includeVdW)
        t = t[:80]
        potentials = [el[:80] for el in potentials]
        vmax = argrelextrema(potentials[0], np.greater)[0]
        vmin = argrelextrema(potentials[0], np.less)[0]
        if len(vmax)==1 and len(vmin)==1:
            return [-min(el[vmax], 0) + el[vmin] for el in potentials][0]
        else:
            if for_minimization:  # We want to push the minimizer towards parameters where there is a trap
                return 0
            return 0
            
    def map_trap_depthv(self, length = 100):
        # length = 1000
        # d = self.get_trap_dephtv(7e-3*6.2,1e-3*8.3)
        depths = np.zeros((length,length))
        x = np.linspace(0, 50e-3,length, endpoint=True)
        y = np.linspace(0, 15e-3,length, endpoint=True)
        for i in range(length):
            for j in range(length):
                depths[i,j] = self.get_trap_dephtv(x[i],y[j])
        return depths,x,y
    
        
    def plot_map_trap_depthv(self,length=100):
        
        depths,x,y = self.map_trap_depthv(length)
        plt.imshow(depths,
                   extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000,
                   origin='lower',
                   aspect=(y.max()-y.min())/(x.max()-x.min()),
                   cmap='jet')
        plt.colorbar()
        plt.title("Trap depth (mK)")
        plt.xlabel("Red detuned light power (mW)")
        plt.ylabel("Blue detuned light power (mW)")
        
    def get_trap_positionv(self, power_b, power_r, includeVdW=True):
        t, potentials = self.potential1dv(power_b, power_r, includeVdW=includeVdW)
        t = t[:80]
        potentials = [el[:80] for el in potentials]
        vmax = argrelextrema(potentials[0], np.greater)[0]
        vmin = argrelextrema(potentials[0], np.less)[0]
        if len(vmax)==1 and len(vmin)==1:
            # print("Minima found")
            return np.real(t[vmin])
        else:
            # print("No mimima was found")
            return 0
        
    def plot_map_pos_depthv(self, length = 100, blue_max_p = 50e-3, red_max_p = 15e-3):
        # length = 1000
        # d = self.get_trap_dephtv(7e-3*6.2,1e-3*8.3)
        pos = np.zeros((length,length))
        x = np.linspace(0, blue_max_p,length, endpoint=True)
        y = np.linspace(0, red_max_p,length, endpoint=True)
        for i in range(length):
            for j in range(length):
                pos[i,j] = self.get_trap_positionv(x[i],y[j])*1e9
                
        plt.imshow(pos,
                    extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000,
                    origin='lower',
                    aspect=(y.max()-y.min())/(x.max()-x.min()),
                    cmap='jet')
        plt.colorbar()
        plt.title("Trap position (nm)")
        plt.xlabel("Red detuned light power (mW)")
        plt.ylabel("Blue detuned light power (mW)")
        
        return pos,x,y
        
    
    def optimize_blue_power(self, pb0=9e-3, pr=1e-3):
        func = lambda pb: self.get_trap_dephtv(pb, pr, for_minimization=True)
        res = so.minimize(func, pb0 , method='Nelder-Mead')  #, bounds = bnds)
        return res
        
    def optimize_powers(self, max_pr=10e-3, n_points=100, plot=True):
        pr0=5e-3 
        pb0=40.1e-3
        blue_powers=np.array([])
        red_powers = np.linspace(pr0,max_pr, n_points)
        depths = np.array([])
        for p in red_powers:
            opt = self.optimize_blue_power(pb0=pb0, pr=p)
            blue_powers = np.append(blue_powers, opt.x[0])
            depths = np.append(depths, opt.fun)
            pb0=opt.x[0]
            # print()
        
        if plot:
            plt.clf()
            plt.plot(red_powers*1e3, -depths)
            plt.xlabel("Power in red detuned beam (mW)\n\nFor each point, the blue detuned beam is optimized\n in order to obtain maximum trap depth")
            plt.ylabel("Trap depth (mK)")
            plt.title("")
            plt.tight_layout()
        return blue_powers, red_powers, depths
    
    def get_powers_one_mK(self, max_pr=10e-3, n_points=100, plot=True):
        b, r, d = self.optimize_powers(max_pr=max_pr, n_points=n_points, plot=plot)
        zero_crossings = np.where(np.diff(np.signbit(d+1)))[0]
        z = zero_crossings
        mr = (r[z+1]-r[z])/(d[z+1]-d[z])
        mb = (b[z+1]-b[z])/(d[z+1]-d[z])
        r0 = r[z]+mr*(d[z]+1)
        b0 = b[z]+mb*(d[z]+1)
        if plot:
            # plt.plot(r[zero_crossings]*1e3, -d[zero_crossings],"xr")
            plt.plot(r0*1e3, 1,"xr")
        return b0[0], r0[0]
        
    
    def get_trap_dephtv_saddle(self, power_b, power_r, mF=0, includeVdW=True, for_minimization=False):
        pot = self.potential2d(power_b, power_r, mF=mF)
        t = self.Simul.y
        depths = []
        for line in pot:
            t = t[:80]
            potential = line[:80]
            vmax = argrelextrema(potential, np.greater)[0]
            vmin = argrelextrema(potential, np.less)[0]
            # print(vmin, vmax)
            if vmin[0]==1:
                vmin = np.delete(vmin, 0)
            if len(vmax)==1 and len(vmin)==1:
                depths.append( -min(potential[vmax], 0) + potential[vmin])
            else:
                depths.append(0)
        return depths

# class square_trap(trap):
#     def __init__(self, 
#                  datafolder_red, 
#                  datafolder_blue,
#                  height = 150e-9,
#                  width = 1000e-9,
#                  wavelengths = [690e-9, 900e-9],
#                  eigenfrequency_r=1,
#                  eigenfrequency_b =1
#                  ):
        
#         super().__init__(self, datafolder_red, datafolder_blue, height, width, wavelengths, eigenfrequency_r, eigenfrequency_b)
        

# class multiple_traps:
#     def __init__(self, filename_red, filename_blue):
#         self.filename_red = filename_red
#         self.filename_blue = filename_blue
#         self.trap11 = self.ft(1,1)
#         self.trap12 = self.ft(1,2)
#         self.trap21 = self.ft(2,1)
#         self.trap22 = self.ft(2,2)
        
#     def ft(self, eb, er):
#         return trap(self.filename_red, 
#                  self.filename_blue,
#                  eigenfrequency_b=eb,
#                  eigenfrequency_r=er,
#                  )


if __name__=="__main__":
    datafolder_red  = os.path.join(os.getcwd(), "datafolder")
    datafolder_blue = os.path.join(os.getcwd(), "datafolder")

    
    height = 150e-9
    width = 1000e-9
    wavelengths = [700e-9, 860e-9]
    eigenfrequency_b=2
    eigenfrequency_r = 2
    aoi = [-width/2, height/2+250e-9, width/2, height/2]    
    
    # t1 = trap(datafolder_red, 
    #           datafolder_blue, 
    #           height = height,
    #           width = width,
    #           wavelengths = wavelengths, 
    #           eigenfrequency_b=eigenfrequency_b,
    #           eigenfrequency_r=eigenfrequency_r,
    #           aoi = aoi,
    #           gridsize = 1000
    #           ) # The csv file contains only 
                # one eigenmode, the 2,2 one
    # b, r = t.get_powers_one_mK()
    # t.plot_potential1dv(10000e-3,1)
    # t.plot_potential2d(1e-3,0)
    # t.plot_map_trap_depthv()
    # t.blue.
    t2 = trap(datafolder_red, 
              datafolder_blue, 
              height = height,
              width = width,
              wavelengths = [690e-9, 850e-9], 
              eigenfrequency_b=eigenfrequency_b,
              eigenfrequency_r=eigenfrequency_r,
              aoi = aoi,
              gridsize = 1000
              )

    # ef1 = t1.blue.efield_b1[4][2,:,:,0]
    ef2 = t2.blue.efield_b1[4][2,:,:,0]
    # plt.imshow(abs(ef1))
    # plt.colorbar()
    # plt.figure()
    plt.imshow(abs(ef2))
    plt.colorbar()
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   
                                                    
                                                   

    # # bp, rp, d = t.optimize_powers(max_pr=10e-3, n_points=100, plot=True)
    # b, r = t.get_powers_one_mW()
    # # t.plot_potential1dv(b,r)
    # # t.plot_potential2d(b,r)
    
    # p = t.get_trap_dephtv_saddle(b,r)


    # potential_m = t.potential2d(b, r, mF = -1)
    # potential_p = t.potential2d(b, r, mF = +1)
    # pot = potential_p - potential_m 
    
    # kB = 1.38064852e-23
    # hbar = 1.0545718e-34
    # gF = -1/2
    # uB = 1.399624624 *1e6 #* hbar * 2 * np.pi  # Hz per Gauss
    
    # factor = kB/hbar/uB/gF/1000
    
    # x = t.Simul.x
    # y = t.Simul.y
    # plt.clf()
    # plt.imshow(-factor*pot.T, 
    #             origin='lower', 
    #             extent=[x.min(),x.max(),y.min(),y.max()], 
    #             cmap="jet_r"
    #             )
    # plt.title("Fictitious B field (G)\nEigenmode %i at $\\lambda=%i$ nm, P=%.1f mW,\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
    #           %(t.eigenfrequency_b, t.blue.wavelength*1e9, b*1e3, t.eigenfrequency_r, t.red.wavelength*1e9, r*1e3))
    # # potential[int(len(self.Simul.y)/2)]=np.zeros(len(potential[0]))
    # # plt.axvline(y=self.Simul.y[int(len(self.Simul.y)/2)], color='red')
    # plt.xlabel("$x$ (m)")
    # plt.ylabel("$y$ (m)")

    # # img = ax.imshow(potential, extent=[Simul.x.min(),Simul.x.max(),Simul.y.min(),Simul.y.max()])
    # plt.colorbar()
    # plt.tight_layout()

    # print(1/(10*0.7*1e6))










