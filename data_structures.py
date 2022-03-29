#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:40:57 2022

@author: jeremy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nanotrappy.utils.physicalunits as pu
import ffd_mode_solver as ffd 


class datastruct:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.wavelength = 0
        self.width, self.height = 0 ,0 
        self.efields = []
        self.neffs = []
        self.filename=""
        
    def __eq__(self, other):
        if self.width!=other.width:
            return False
        if self.height!=other.height:
            return False
        if len(self.x)!=len(other.x):
            return False
        if len(self.y)!=len(other.y):
            return False
        if len(self.z)!=len(other.z):
            return False
        return True
    
    def export_to_eigenfolders(self, redorblue=''):
        options = ['r1b1','r1b2','r2b1','r2b2',]
        o1 = self.efields[0]
        o2 = self.efields[1]
        f1 = np.array([o1[0], o1[2], o1[1], o1[3], o1[4]], dtype=object)
        f2 = np.array([o2[0], o2[2], o2[1], o2[3], o2[4]], dtype=object)
        for i, o in enumerate(options):
            folder = os.path.join(os.path.dirname(self.filename),"eigen_"+o)
            fileN = os.path.basename(self.filename)
            os.makedirs(folder, exist_ok=True)
            fname1 = os.path.join(folder, fileN+"_eigen1.npy")
            fname2 = os.path.join(folder, fileN+"_eigen2.npy")
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

        
    
    def plot_intensity(self, eigenfrequency, plot=True, cmap='jet'):
        dat = self.efields[eigenfrequency]
        x = dat[1]
        y = dat[2]
        e_field = dat[4]
        # intensity = 0.5*pu.cc*np.sqrt(1)*(abs(e_field)**2).sum(axis=0)[:,:,0] #  TODO: insert effective refractive index here
        intensity = (abs(e_field)**2).sum(axis=0)[:,:,0] #  TODO: insert effective refractive index here
        
        if plot:
            plt.clf()
            plt.imshow(intensity, 
                       origin='lower', 
                       extent=[x.min()*1e9, x.max()*1e9, y.min()*1e9, y.max()*1e9], 
                       cmap=cmap
                       )
            # plt.xlabel("x (nm)")
            # plt.ylabel("y (nm)")
            plt.title("Intensity of eigenmode %i at $\\lambda=%i$ nm\nHeight=%i nm, Width=%i nm"
                      %(eigenfrequency, self.wavelength*1e9, self.height*1e9, self.width*1e9))
            plt.colorbar()
            plt.tight_layout()
        return intensity
    
    
class comsol_data(datastruct):
    def __init__(self, filename, n_eigenfrequencies=2, aoi=[], power = 0):
        super().__init__()
        self.aoi = aoi
        self.convert = np.vectorize(self.convert_s)
        
        self.filename = filename
        wavelength = str(os.path.split(filename)[1].split('_')[4].split('.csv')[0])
        self.wavelength=int(wavelength)*1e-10 if len(wavelength)==4 else int(wavelength)*1e-9
        self.width, self.height = self.get_waveguide_dimensions()
        # self.efield = [self.get_efield_from_csv(i) for i in range(n_eigenfrequencies)]   #TODO, this line instead of the next, in principle
        self.efields = [self.get_efield_from_csv(0, power=power), self.get_efield_from_csv(0, power=power)]
        self.neffs = [self.get_neff_from_csv(0), self.get_neff_from_csv(0)]
        
        # TODO: add get_neff
        
        self.x = np.real(self.efields[0][1])
        self.y = np.real(self.efields[0][2])
        self.z = np.real(self.efields[0][3])


    def convert_s(self, s):
        cn = complex(str(s).replace("i","j"))
        return cn
    
    def get_wavelength(self):
        pass
    
    def get_waveguide_dimensions(self):
        n = os.path.split(self.filename)[1].split('H_')[1].split('_W_')
        height = n[0]
        width = n[1].split('_')[0]
        return float(width)*1e-9, float(height)*1e-9
    
    def get_neff_from_csv(self, eigenfrequency):
        directory, filen = os.path.split(self.filename)
        filen = filen.split("_")
        newstring = "H_"+filen[1]+"_W_"+filen[3]+"_P.csv"
        power_data = pd.read_csv(os.path.join(directory, newstring), header=4).to_numpy()
        neff = 0
        for line in power_data:
            try:
                np.testing.assert_approx_equal(line[0], self.wavelength, significant=3)
                neff = line[1]
            except AssertionError:
                pass
        return neff
    
    def get_efield_from_csv(self, eigenfrequency, power=0):
        df = pd.read_csv(self.filename, header=8)
        # self.df = df
        data = np.array([self.convert(el) for el in df.to_numpy().transpose()])
        ex = data[3+(eigenfrequency)*5]
        ey = data[4+(eigenfrequency)*5]
        ez = data[5+(eigenfrequency)*5]
        # power = np.average(np.array(data[3+(eigenfrequency-1)*5]))
        directory, filen = os.path.split(self.filename)
        filen = filen.split("_")
        newstring = "H_"+filen[1]+"_W_"+filen[3]+"_P.csv"
        power_data = pd.read_csv(os.path.join(directory, newstring), header=4).to_numpy()
        if power==0:
            for i, line in enumerate(power_data):
                try:
                    np.testing.assert_approx_equal(line[0], self.wavelength, significant=3)
                    if "TE" in self.filename and i%2==0:
                        power = line[-1]
                    elif "TM" in self.filename and i%2==1:
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
        E_field_normalized = E_field/np.sqrt(power)
        E_field_normalizedT = E_field_normalized.transpose(0,2,1,3)
        formated_array = np.array([self.wavelength, x,y,z, E_field_normalizedT], dtype=object)
        if self.aoi!=[]:
            ef = formated_array
            x = ef[1] 
            y = ef[2]
            z = ef[3]
            e = ef[4]
            ex = e[0]
            ey = e[1]
            ez = e[2] 
            if self.aoi != []:
                xtruth = (x>=self.aoi[0]) * (x<=self.aoi[2])
                ytruth = (y<=self.aoi[1]) * (y>=self.aoi[3])
            else:
                xtruth = (x==x)
                ytruth = (y==y)
            xnew = x[xtruth]
            ynew = y[ytruth] 
            exnew = np.array([line[xtruth] for line in ex[ytruth]])#.transpose(1,0,2)
            eynew = np.array([line[xtruth] for line in ey[ytruth]])#.transpose(1,0,2)
            eznew = np.array([line[xtruth] for line in ez[ytruth]])#.transpose(1,0,2)
            efields_restricted=np.array([ef[0], xnew, ynew, z, np.array([exnew, eynew, eznew])], dtype=object)
            return efields_restricted
        return formated_array

    
class emepy_data(datastruct):
    def __init__(self, data_folder, height, width, wavelength, gridsize, aoi, x=[], y=[], n_eigenfrequencies=2):
        super().__init__()
        self.x=x
        self.y=y
        self.height, self.width = height, width
        self.gridsize = gridsize
        self.aoi = aoi
        self.n_eigenfrequencies = n_eigenfrequencies
        self.wavelength = wavelength
        self.data_folder = data_folder
        self.filename = os.path.join(data_folder,"H_%i_W_%i_%i"%(height*1e9, width*1e9, wavelength*1e10))
        self.efields, self.neffs = self.get_efield()
        self.x = np.real(self.efields[0][1])
        self.y = np.real(self.efields[0][2])
        self.z = np.real(self.efields[0][3])

    def get_efield(self):
        simu = ffd.simulation(
            wavelength=self.wavelength, 
            height=self.height, 
            width=self.width,
            gridsize=self.gridsize,
            x=self.x,
            y=self.y,
            nmodes=self.n_eigenfrequencies,)
        self.simu = simu
        efields = [simu.formated_data[eig] for eig in range(self.n_eigenfrequencies)]
        neffs = [simu.neff[eig] for eig in range(self.n_eigenfrequencies)]
        
        # Now, restrict to area of interest aoi
        efields_restricted = []
        for ef in efields:
            x = ef[1] 
            y = ef[2]
            z = ef[3]
            e = ef[4]
            ex = e[0]
            ey = e[1]
            ez = e[2] 
            if self.aoi != []:
                xtruth = (x>=self.aoi[0]) * (x<=self.aoi[2])
                ytruth = (y<=self.aoi[1]) * (y>=self.aoi[3])
            else:
                xtruth = (x==x)
                ytruth = (y==y)
            xnew = x[xtruth]
            ynew = y[ytruth] 
            exnew = np.array([line[xtruth] for line in ex[ytruth]])#.transpose(1,0,2)
            eynew = np.array([line[xtruth] for line in ey[ytruth]])#.transpose(1,0,2)
            eznew = np.array([line[xtruth] for line in ez[ytruth]])#.transpose(1,0,2)
            efields_restricted.append(np.array([ef[0], xnew, ynew, z, np.array([exnew, eynew, eznew])], dtype=object))
        
        return efields_restricted, neffs

        
        

if __name__=="__main__":
    data_folder = os.path.join(os.getcwd(), "datafolder")
    # o = comsol_data(data_folder+"/H_150_W_1000_7000.csv")
    height = 150e-9
    width = 1e-6
    # aoi = [-width/2, height/2+250e-9, width/2, height/2]
    aoi = []
    # aoi=[]
    # o_red = emepy_data(data_folder, height, width, 850e-9, 1000, aoi, [], [], 2)
    # o_blue = emepy_data(data_folder, height, width, 690e-9, 1000, aoi, [], [], 2)
    # o_blue.export_to_eigenfolders("blue")
    # plt.figure()
    # o_red.plot_intensity(1)
    # o_blue.plot_intensity(0)
    # plt.figure()
    # o_blue.simu.plot_field(1)
    # print(o.filename)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    