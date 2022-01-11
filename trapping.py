#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:41:36 2022

@author: jeremy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nanotrappy as nt
import nanotrappy.utils.physicalunits as pu
from nanotrappy.utils import vdw
import warnings
import scipy.optimize as so
from scipy.signal import argrelextrema
import matplotlib.patches as mpl_patches
from data_structures import emepy_data, comsol_data


class trap:
    def __init__(self, blue, red, eigen_r=0, eigen_b =0,):
        self.blue = blue
        self.red = red
        if red!=blue:
            raise Exception("Blue and Red don't have the same parameters")
        self.eigen_r = eigen_r
        self.eigen_b = eigen_b
        blue.export_to_eigenfolders("blue")
        red.export_to_eigenfolders("red")


        self.red_beam = nt.trapping.beam.BeamPair(self.red.wavelength, 1 , self.red.wavelength, 1 )
        self.blue_beam = nt.trapping.beam.Beam(self.blue.wavelength, "f", 1 )
        # self.blue_beam2 = nt.trapping.beam.Beam(self.blue.wavelength+30e9, "b", 1 )
        self.trap = nt.trapping.trap.Trap_beams(self.blue_beam, self.red_beam)
        self.syst = nt.trapping.atomicsystem.atomicsystem(nt.Rubidium87(), "5S1/2", pu.f2)
        self.surface = vdw.PlaneSurface(normal_axis=nt.trapping.geometry.AxisX(), 
                                        normal_coord=self.red.height/2.0)
        self.Simul = nt.trapping.simulation.Simulation(
            self.syst,
            nt.SiN(),
            self.trap,
            os.path.join(os.path.dirname(red.filename),'eigen_r%ib%i'%(eigen_r+1,eigen_b+1)),
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

    def get_potential2d(self, power_b=40e-3, power_r=5e-3, mF = 0, remove_center=False, includeVdW=True):
        pot = power_b*self.potentials[0,:,:,mF+2]+power_r*self.potentials[1,:,:,mF+2]#+power_b/2*self.potentials[2,:,:,mF+2]
        if includeVdW:
            pot+=self.Simul.CP
        x = self.Simul.x
        y = self.Simul.y
        width = self.red.width
        height = self.blue.height
        if remove_center:
            for i in range(len(pot)):
                for j in range(len(pot[0])):
                    if abs(x[i])<(height/2) and abs(y[j])<(width/2):
                        pot[i,j]=0
        return pot


    def plot_potential2d(self, power_b=1, power_r=1, mF = 0, remove_center=True, cmap='jet_r', includeVdW=True):
        potential = self.get_potential2d(power_b, power_r, mF, remove_center=remove_center, includeVdW=includeVdW)
        if remove_center:
            depth = -self.get_trap_depht_and_position(power_b, power_r)[0][mF+2]
            potential[abs(potential)>depth*1.02]=0
            potential[potential>0.001]=0
        x = np.real(self.Simul.x)
        y = np.real(self.Simul.y)
        plt.clf()
        plt.imshow(potential, 
                   origin='lower', 
                   extent=[y.min(),y.max(),x.min(),x.max()], 
                   cmap=cmap
                   )
        plt.title("Temperature (mK)\nEigenmode %i at $\\lambda=%i$ nm, P=%.1f mW,\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                  %(self.eigen_b, self.blue.wavelength*1e9, power_b*1e3, 
                    self.eigen_r, self.red.wavelength*1e9, power_r*1e3))
        plt.xlabel("$x$ (m)")
        plt.ylabel("$y$ (m)")
        plt.colorbar()
        plt.tight_layout()
        
        
    def get_potential1d(self, power_b=40e-3, power_r=6e-3, n_col=-1, includeVdW=True):
        n_col = int(len(self.Simul.y)/2) if n_col==-1 else n_col
        truth = np.where(self.Simul.x>self.red.height/2)
        potentials1d = []
        for i in range(-2,2+1):
            potential = power_b*self.potentials[0,:,:,i+2]+power_r*self.potentials[1,:,:,i+2]
            potentials1d.append(potential[:,n_col][truth])
        if includeVdW:
            potentials1d = [el+self.Simul.CP[:,n_col][truth] for el in potentials1d]
        return self.Simul.x[truth], potentials1d

    def plot_potential1d(self, power_b=40e-3, power_r=6e-3, includeVdW=True):  # TODO: currently only for mF=2, make more general
       plt.clf()
       t, potentials = self.get_potential1d(power_b, power_r, includeVdW=includeVdW)
       # for i, pot in enumerate(potentials):
       pot = potentials[2]
       plt.plot((t-self.red.height/2)*1e9, pot)# , label="$m_f=%i$"%(i-2))
       plt.axvline(x=0, color='black')
       plt.title("Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                 %(self.eigen_b, self.blue.wavelength*1e9,power_b*1e3, self.eigen_r, self.red.wavelength*1e9,power_r*1e3))
       plt.xlabel("Distance from surface (nm)")
       plt.ylabel("Potential (mK)")
       ax = plt.gca()
       depth = self.get_trap_depht_and_position(power_b, power_r)[0][2]
       if depth!=0:
           t, potentials = self.get_potential1d(power_b, power_r, includeVdW=includeVdW)
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

    def optimize_blue_power(self, pb0=9e-3, pr=1e-3):
        func = lambda pb: self.get_trap_depht_and_position(pb, pr, for_minimization=True)[0][2]
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
            plt.plot(r0*1e3, 1,"xr")
        return b0[0], r0[0]

    def get_trap_depht_and_position(self, power_b, power_r, includeVdW=True, for_minimization=False):  
        t, potentials = self.get_potential1d(power_b, power_r, includeVdW=includeVdW)
        t = t[:80]
        potentials = [el[:80] for el in potentials]
        vmax = [argrelextrema(pot, np.greater)[0] for pot in potentials]
        vmin = [argrelextrema(pot, np.less)[0] for pot in potentials]
        depths = []
        positions = []
        for i in range(len(vmax)):
            if len(vmax[i])==1 and len(vmin[i])==1:
                depths.append(-min(potentials[i][vmax[i]], 0) + potentials[i][vmin[i]])
                positions.append(np.real(t[vmin[i]]))
            elif for_minimization:
                depths.append(0)   
                positions.append(0)
            else:
                depths.append(0)
                positions.append(0)
        return depths, positions 

    def map_trap_depth_pos(self, length = 100, mF=0, plot=False):
        depths = np.zeros((length,length))
        positions = np.zeros((length,length))
        x = np.linspace(0, 50e-3,length, endpoint=True)
        y = np.linspace(0, 15e-3,length, endpoint=True)
        for i in range(length):
            for j in range(length):
                depth, position = self.get_trap_depht_and_position(x[i],y[j])
                depths[i,j], positions[i,j] = depth[mF+2], position[mF+2]
        if plot:
            plt.imshow(depths,
                       extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000,
                       origin='lower',
                       aspect=(y.max()-y.min())/(x.max()-x.min()),
                       cmap='jet')
            plt.colorbar()
            plt.title("Trap depth (mK)")
            plt.xlabel("Red detuned light power (mW)")
            plt.ylabel("Blue detuned light power (mW)")
            plt.figure()
            plt.imshow(positions,
                       extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000,
                       origin='lower',
                       aspect=(y.max()-y.min())/(x.max()-x.min()),
                       cmap='jet')
            plt.colorbar()
            plt.title("Trap position (nm)")
            plt.xlabel("Red detuned light power (mW)")
            plt.ylabel("Blue detuned light power (mW)")
        return depths, positions, x, y        


if __name__=="__main__":
    data_folder = os.path.join(os.getcwd(), "datafolder")
    height = 150e-9
    width = 1e-6
    aoi = [-width/2, height/2+500e-9, width/2, height/2]
    o_red =  emepy_data(data_folder, height, width, 850e-9, 500, aoi, [], [], 2)
    o_blue = emepy_data(data_folder, height, width, 690e-9, 500, aoi, [], [], 2)
    
    # o_red = comsol_data(data_folder+"/H_150_W_1000_860.csv")
    # o_blue = comsol_data(data_folder+"/H_150_W_1000_700.csv")
    t = trap(o_blue, o_red, 0,0)
    t.red.plot_intensity(0)
    plt.figure()
    t.blue.plot_intensity(0)
    # t.plot_potential1d(47e-3,6e-3)
    # d,p, xx,yy = t.map_trap_depth_pos(plot=True)
    # b,r = t.get_powers_one_mK()
    # t.plot_potential2d(b,r)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    