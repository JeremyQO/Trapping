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
            cp = self.Simul.CP
            # cp[cp<-1e6]=0
            pot+=cp
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

    def plot_potential2d(self, power_b=5e-2, power_r=6.57e-3, mF = 0, remove_center=True, cmap='jet_r', includeVdW=True):
        potential = self.get_potential2d(power_b, power_r, mF, remove_center=remove_center, includeVdW=includeVdW)
        if remove_center:
            depth = -self.get_trap_depht_and_position(power_b, power_r)[0][mF+2]
            # depth = 1.02
            potential[abs(potential)>depth*1.02]=0
            potential[potential>0.001]=0
        x = np.real(self.Simul.x)
        y = np.real(self.Simul.y)
        plt.clf()
        im = plt.imshow(potential, 
                   origin='lower', 
                   extent=[y.min(),y.max(),x.min(),x.max()], 
                   cmap=cmap
                   )
        plt.title("Temperature (mK)\nEigenmode %i at $\\lambda=%i$ nm, P=%.1f mW,\n Eigenmode %i at $\\lambda=%i$ nm, P=%.1f mW"
                  %(self.eigen_b, self.blue.wavelength*1e9, power_b*1e3, 
                    self.eigen_r, self.red.wavelength*1e9, power_r*1e3))
        plt.xlabel("$x$ (m)")
        plt.ylabel("$y$ (m)")
        plt.colorbar(im ,fraction=0.046, pad=0.04)
        plt.tight_layout()
        
    def get_potential1d(self, power_b=40e-3, power_r=6e-3, n_col=-1, includeVdW=True):
        n_col = int(len(self.Simul.y)/2) if n_col==-1 else n_col
        truth = np.where(self.Simul.x>self.red.height/2)
        potentials1d = []
        for i in range(-2,2+1):
            potential = power_b*self.potentials[0,:,:,i+2]+power_r*self.potentials[1,:,:,i+2]
            potentials1d.append(potential[:,n_col][truth])
        if includeVdW:
            cp = self.Simul.CP[:,n_col][truth]
            # cp[cp<-1e6]=0
            potentials1d = [el+cp for el in potentials1d]
        return self.Simul.x[truth], potentials1d

    def plot_potential1d(self, power_b=50e-3, power_r=6e-3, includeVdW=True):  # TODO: currently only for mF=2, make more general
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
           vmin = argrelextrema(potentials[0], np.less)[0][-1]
           position = t[vmin]
           plt.plot((position-self.red.height/2)*1e9, potentials[0][vmin], 'xr')
           # plt.xlabel("Distance from surface (nm)\n\n\nTrap depth %.0f uK\nTrap position %.0f nm"%(-depth*1e3, (position-self.red.height/2)*1e9))#, xy, args, kwargs)
           minn = potentials[0][vmin]
           mminn = minn if minn >-100 else -30
           ax.set_ylim([mminn*1.05, 50e-3])
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
        pr0=3e-3 
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
            plt.plot(blue_powers*1e3, -depths,'blue', label="blue")
            plt.plot(red_powers*1e3, -depths,'red', label="red")
            plt.xlabel("Power in dipole beams (mW)\n\nFor each point, the blue detuned beam is optimized\n in order to obtain maximum trap depth")
            plt.ylabel("Trap depth (mK)")
            plt.legend()
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
            plt.plot(b0*1e3, 1,"xb")
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
            if len(vmax[i])==1 and len(vmin[i])>=1:
                depths.append(-min(potentials[i][vmax[i]], 0) + potentials[i][vmin[i][-1]])
                positions.append(np.real(t[vmin[i][-1]]))
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
        x = np.linspace(0, 60e-3,length, endpoint=True)
        y = np.linspace(0, 7.5e-3,length, endpoint=True)
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
            plt.tight_layout()
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
            plt.tight_layout()
        return depths, positions, x, y        


    def get_trap_frequency(self, power_b=55e-3, power_r=5.5e-3, n_line=-1, plot=False):  # TODO: make this work
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


if __name__=="__main__":
    data_folder = os.path.join(os.getcwd(), "datafolder")
    data_folder = os.path.join(os.getcwd(), "datafolder/690-850")
    height = 150e-9
    width = 1e-6
    aoi = [-width/2, height/2+500e-9, width/2, height/2]
    
    # o_red  = comsol_data(data_folder+"/H_150_W_1000_850.csv", aoi=aoi)
    # o_blue = comsol_data(data_folder+"/H_150_W_1000_690.csv", aoi=aoi)
    o_blue = emepy_data(data_folder, height, width, 690e-9, 1000, aoi, [], [], 2)
    o_red =  emepy_data(data_folder, height, width, 850e-9, 1000, aoi, [], [], 2)
    t = trap(o_blue, o_red, 1,1)
    # b,r = t.get_powers_one_mK(plot=False)
    # plt.figure()
    # t.plot_potential1d(b,5.4e-3)    
    # t.plot_potential2d(b,r)
    # d,p,x,y = t.map_trap_depth_pos(1000, plot=True)
    # string = "Height = 150nm. Width = 1000 nm. LOCAs. Blue wavelength = 690 nm. Red wavelength = 850 nm. Rubidium 87 in F=2. Atom trapped bove the waveguide. Powers are expressed in watts "
    # np.savez_compressed('depth_pos', depth=d, position=p, blue_power=x, red_power=y, readme=string)
    # t.plot_potential1d(50e-3,0.2e-3)
    # from scipy import interpolate
    # depth = interpolate.interp2d(x, y, d, kind='linear', fill_value=0)
    # pos = interpolate.interp2d(x, y, p, kind='linear', fill_value=0)
    # xx = np.linspace(x[0], x[-1],2000)
    # yy = np.linspace(y[0], y[-1], 2000)
    # dd = depth(xx,yy)    
    # pp = pos(xx,yy)
    # plt.imshow(pp)    
    # plt.imshow(dd)    
    # plt.figure()
    # plt.clf()
    # # plt.plot(y*1e3,d[917], label='Constant Blue')
    # plt.plot(y*1e3,d[917], label='Constant Blue')
    # plt.xlabel("Power of red detuned laser (mW)", )
    # plt.ylabel("Trap depth (mK)", )
    # plt.legend()
    # plt.tight_layout()
    # plt.figure()
    # plt.plot(y[:-1]*1e3, np.diff(d[917])/np.diff(y*1e3))
    # plt.plot(x[:-1]*1e3, np.diff(d[:,745])/np.diff(x*1e3))
    # plt.figure()
    # plt.clf()
    # plt.plot(y*1e3,(p[917]-75e-9)*1e9, label='Constant Blue')
    # plt.xlabel("Power of red detuned laser (mW)", )
    # plt.ylabel("Trap position (nm)", )
    # plt.legend()
    # plt.tight_layout()
    
    
    # plt.figure()
    # plt.clf()
    # plt.plot(x*1e3,(p[:,745]-75e-9)*1e9, label='Constant Red')
    # plt.title("Distance between minima of trap and waveguide surface")
    # plt.xlabel("Power of blue detuned laser (mW)", )
    # plt.ylabel("Trap position (nm)", )
    # plt.legend()
    # plt.tight_layout()
    
    
    # yy = (p[:,745]-75e-9)*1e9
    # xx = x*1e3
    # print(30/(57-49.5))
    
    # # plt.figure()
    # # plt.clf()
    # # plt.plot(x[:-1]*1e3,np.diff(p[:,745]*1e9)/np.diff(x*1e3), label='Constant Red')
    # # plt.xlabel("Power of blue detuned laser (mW)", )
    # # plt.legend()
    # # plt.tight_layout()
    
    # np.diff(p[:,745]*1e9)/np.diff(x*1e3)
    
    # o_red =  emepy_data(data_folder, height, width, 850e-9, 1000, aoi, [], [], 2)
    # o_red2 = comsol_data(data_folder2+"/H_150_W_1000_850.csv", aoi=aoi)
    # e1r = np.real(o_red.efields[0][4][0,:,:,0])
    # e2r = np.real(o_red.efields[0][4][1,:,:,0])
    # e3r = np.imag(o_red.efields[0][4][2,:,:,0])
    # x = np.real(o_red.efields[0][1])
    # y = np.real(o_red.efields[0][2])
    # from scipy import interpolate
    # f1r = interpolate.interp2d(x, y, e1r, kind='linear', fill_value=0)
    # f2r = interpolate.interp2d(x, y, e2r, kind='linear', fill_value=0)
    # f3r = interpolate.interp2d(x, y, e3r, kind='linear', fill_value=0)

        
    # e1r2 = np.real(o_red2.efields[0][4][0,:,:,0])
    # e2r2 = np.real(o_red2.efields[0][4][1,:,:,0])
    # e3r2 = np.imag(o_red2.efields[0][4][2,:,:,0])
    # x2 = np.real(o_red2.efields[0][1])
    # y2 = np.real(o_red2.efields[0][2])
    
    # e1r1 = f1r(x2,y2)
    # e2r1 = f2r(x2,y2)
    # e3r1 = f3r(x2,y2)
    # e1r1[np.isnan(e1r1)]=0
    # e2r1[np.isnan(e2r1)]=0
    # e3r1[np.isnan(e3r1)]=0
    
    
    # aoi = []
    # o_blue = emepy_data(data_folder, height, width, 690e-9, 1000, aoi, [], [], 2)
    # o_red =  emepy_data(data_folder, height, width, 850e-9, 1000, aoi, [], [], 2)

    # o_red  = comsol_data(data_folder+"/H_150_W_1000_850.csv", aoi=aoi)
    # o_blue = comsol_data(data_folder+"/H_150_W_1000_690.csv", aoi=aoi)
    # t = trap(o_blue, o_red, 0, 0)
    # b,r = (0.049311874391455214, 0.006579567451118707*0)
    # o_blue.plot_intensity(0)
    # print(o_blue.efields[0][4].shape)
    # t.red.plot_intensity(1)
    # plt.figure()
    # t.blue.plot_intensity(1)
    # t.plot_potential1d(40e-3,3e-3)
    # d,p, xx,yy = t.map_trap_depth_pos(plot=True)
    # b,r = t.get_powers_one_mK()
    # plt.figure()
    # b=0.0542630823379978
    # r = 0.0055024679348216025
    # t.plot_potential1d(b,r)
    # t.plot_potential2d(0,r, includeVdW=False, remove_center=False)

    # t.plot_potential2d(75e-3,7e-3, includeVdW=True, remove_center=True)    
    # t.get_trap_depht_and_position(65e-3,7e-3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    # files = ["H_150_W_1000_6900_eigen1_1000c.npz","H_150_W_1000_6900_eigen2_1000c.npz",
    #           "H_150_W_1000_8500_eigen1_1000c.npz", "H_150_W_1000_8500_eigen2_1000c.npz"]
    # files = ["H_150_W_1000_8500_eigen2_1000c.npz"]
    # data_folder = os.path.join(os.getcwd(), "datafolder")
    # fs = np.array([os.path.join(data_folder, f) for f in files])
    # for i in range(1):
    #     d = np.load(fs[i], allow_pickle=True)
    #     neff = d['neff']
    #     fa = d['formatted_array']
    #     mu0 = 4*np.pi*1e-7
    #     c = 299792458    
    #     fa4 = fa[4]* np.sqrt(mu0 * c)
    #     fa[4]=fa4
    #     print(fs[i].replace("c","").replace(".npz",""))
    #     np.savez_compressed(fs[i].replace("c.npz",""), formatted_array=fa, neff=neff)

    
    
    
    
    
    
    
    
    
    
    
    