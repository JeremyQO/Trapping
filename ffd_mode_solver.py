"""Fully vectorial finite-difference mode solver example."""

import numpy as np
import EMpy
import pylab
import matplotlib.pyplot as plt
import time 
import os

class simulation:
    def __init__(self, wavelength=780e-9, height=0.15e-6, width=1e-6, x=[], y=[], gridsize=250, nmodes=2, loadsaved = True):

        self.wavelength = wavelength
        self.height = height
        self.width = width
        self.gridsize = gridsize 
        self.datafolder = os.path.join(os.getcwd(), "datafolder")
        
        if len(x)==0:
            # dense = np.linspace(-1e-6, 1e-6, 150)
            # not_dense_pos = np.linspace(1e-6, 2.25e-6, 30)
            # not_dense_neg = np.linspace(-2.25e-6, -1e-6, 30) 
            # self.x = np.unique(np.concatenate((not_dense_neg, dense, not_dense_pos)))
            # self.y = np.unique(np.concatenate((not_dense_neg, dense, not_dense_pos)))
            self.x = np.linspace(-2.25e-6, 2.25e-6, gridsize)
            self.y = np.linspace(-2.25e-6, 2.25e-6, gridsize)
        else:
            self.x = np.array(x)
            self.y = np.array(y)
        self.neigs = nmodes
        self.tol = 1e-8
        self.boundary = '0000'
        
        if (not self.were_already_simulated()) or (not loadsaved):
            start_time = time.time()
            print("Calculating the field with gridsize %i..."%(self.gridsize))
            self.solver = EMpy.modesolvers.FD.VFDModeSolver(self.wavelength, 
                                                            self.x, 
                                                            self.y, 
                                                            self.epsfunc, 
                                                            self.boundary).solve(self.neigs, self.tol)
            self.save_simulations()
            self.formated_data = [self.get_formatted_array(i) for i in range(self.neigs)]
            self.neff = [self.get_neff(i) for i in range(self.neigs)]
            [self.is_guided(i) for i in range(self.neigs)]
            print("Done. \nRunning time: %.2f seconds."%(time.time() - start_time))
            # print("Simulation was saved in file %s"%)
        else:
            print("\nLoaded simulation from existing files. \nWavelength = %.2f nm"%(1e9*self.wavelength))
            self.formated_data, self.neff = np.array([self.load_simulation(i) for i in range(self.neigs)], dtype=object).T
            [self.is_guided(i) for i in range(self.neigs)]

            
       

    def epsfunc(self, x, y):
        """Return a matrix describing a 2d material.
    
        :param x_: x values
        :param y_: y values
        :return: 2d-matrix
        """
        xx, yy = np.meshgrid(x, y)
        n_vacuum = np.where(np.abs(xx.T)==np.abs(xx.T), 1, 0)
        n_SiO2 = np.where(yy.T < -self.height/2+50e-9,  # added 50 nm of silica for the LOCAs geometry
                           self.get_n_lambda(self.wavelength, 'SiO2')**2,
                           n_vacuum)
        
        n_SiN = np.where((np.abs(xx.T) <= self.width/2) * (np.abs(yy.T) <= self.height/2),
                         self.get_n_lambda(self.wavelength, 'SiN')**2,
                         n_SiO2
                         )
        return n_SiN 
    
    def plot_n(self):
        plt.imshow(np.sqrt(self.epsfunc(self.x,self.y).T), 
                   extent=[self.y.min(),self.y.max(),self.x.min(),self.x.max()],
                   origin='lower',
                   aspect=(self.y.max()-self.y.min())/(self.x.max()-self.x.min()))
        
    def get_Efield(self, n_mode=0):
        if hasattr(self, "solver"):
            Ex = np.transpose(self.solver.modes[n_mode].get_field('Ex', self.x, self.y))
            Ey = np.transpose(self.solver.modes[n_mode].get_field('Ey', self.x, self.y))
            Ez = np.transpose(self.solver.modes[n_mode].get_field('Ez', self.x, self.y))
            return np.array([Ex[..., np.newaxis], Ey[..., np.newaxis], Ez[..., np.newaxis]])
        return self.formated_data[n_mode][4]
    
    def get_Hfield(self, n_mode=0):
        Hx = np.transpose(self.solver.modes[n_mode].get_field('Hx', self.x, self.y))
        Hy = np.transpose(self.solver.modes[n_mode].get_field('Hy', self.x, self.y))
        Hz = np.transpose(self.solver.modes[n_mode].get_field('Hz', self.x, self.y))
        return np.array([Hx[..., np.newaxis], Hy[..., np.newaxis], Hz[..., np.newaxis]])
    
    
    def get_fictitious_B(self, n_mode):
        from nanotrappy import atomicsystem
        from arc import Rubidium87
        import arc
        # import arc
        
        atom = Rubidium87()
        atomic_system = atomicsystem(atom,"5S1/2",f=1)
        atomic_system.set_alphas(self.wavelength)
        # s = atomic_system.alpha_scalar(self.wavelength) 
        v = atomic_system.alpha_vector(self.wavelength)
        # print(v)
        pol = arc.calculations_atom_single.DynamicPolarizability(atom, 5, 0, 1/2)
        # atomic_system.level_mixing(0,1,0)
        # h = 6.62607015e-34
        # print(s/h)
        # gsp = 0.0794
        # alp = gsp*( (1/795e-9)**2/((1/795e-9)**2 - (1/self.wavelength)**2))
        # print(alp )
        # print(alp/(s/h))

        uB = 9.27400949e-24 # J/T
        v = -1.3256023075258553e-36
        gJ = 2.00233113
        gI = -0.0009951414
        F = 1
        I = 3/2
        J = 1/2
        gF = gJ*(F*(F+1)-I*(I+1)+J*(J+1))/(2*F*(F+1)) + gI*(F*(F+1)+I*(I+1)-J*(J+1))/(2*F*(F+1))  # approx -0.5
        
        e = self.get_Efield(n_mode)
        Gauss = 1e-4  # conversion from tesla
        fictB = 1j*np.cross(np.conj(e), e, axis=0) * v/(8*uB*gF*F) / Gauss
        return fictB
    
    def plot_fictitious_By(self, nmode, remove_center=True):
        Bx, By, Bz = self.get_fictitious_B(nmode)
        By = By[:,:,0]
        x = self.x
        y = self.y 
        if remove_center:
            for i in range(len(Bx)):
                for j in range(len(Bx[0])):
                    width = self.width
                    height = self.height
                    if abs(x[i])<(width/2) and abs(y[j])<(height/2):
                        By[j,i]=0
                                
        width = self.width*1000000
        height = self.height*1000000
        
        plt.imshow(np.real(By),
                    extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000000,
                    origin='lower',
                    aspect=(y.max()-y.min())/(x.max()-x.min()),
                    cmap='jet',
                    # interpolation='spline36',
                    )
        plt.title('Fictitious $B_y$')
        plt.xlabel("$y$ position ($\mu m$)")

        if not remove_center:
            plt.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                      [-height/2, -height/2, height/2, height/2,-height/2],'r')
        plt.colorbar()
        plt.tight_layout()
    
    def plot_fictitious_B(self, nmode, remove_center=True):
        Bx, By, Bz = self.get_fictitious_B(nmode)
        Bx = Bx[:,:,0]
        By = By[:,:,0]
        Bz = Bz[:,:,0]


        x = self.x
        y = self.y 
        
        
        if remove_center:
            for i in range(len(Bx)):
                for j in range(len(Bx[0])):
                    width = self.width
                    height = self.height
                    if abs(x[i])<(width/2) and abs(y[j])<(height/2):
                        Bx[j,i]=0
                        By[j,i]=0
                        Bz[j,i]=0
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))    
                        
        width = self.width*1000000
        height = self.height*1000000
        
        minmin = np.min([np.min(abs(Bx)), np.min(np.real(By))])
        maxmax = np.max([np.max(abs(Bx)), np.max(np.real(By))])

        axes[0].imshow(abs(Bx),
                    vmin=minmin, vmax=maxmax,
                    extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000000,
                    origin='lower',
                    aspect=(y.max()-y.min())/(x.max()-x.min()),
                    cmap='jet')
        axes[0].set_title('Fictitious $B_x$')
        axes[0].set_xlabel("$y$ position ($\mu m$)")
        
        imm = axes[1].imshow(np.real(By),
                    vmin=minmin, vmax=maxmax,
                    extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000000,
                    origin='lower',
                    aspect=(y.max()-y.min())/(x.max()-x.min()),
                    cmap='jet')
        axes[1].set_title('Fictitious $B_y$')
        axes[1].set_xlabel("$y$ position ($\mu m$)")

        
        # axes[2].imshow(abs(Bz),
        #             extent=np.array([y.min(),y.max(),x.min(),x.max()])*1000000,
        #             origin='lower',
        #             aspect=(y.max()-y.min())/(x.max()-x.min()),
        #             cmap='jet')
        # axes[2].set_title('Fictitious $B_z$')
        # axes[1].set_xlabel("$y$ position ($\mu m$)")

        
        fig.subplots_adjust(right=1.4)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
        fig.colorbar(imm, cax=cbar_ax)


        if not remove_center:
            for ax in axes:
                ax.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                          [-height/2, -height/2, height/2, height/2,-height/2],'r')

        pylab.show()
        fig.tight_layout()

    
    def get_neff(self, n_mode=0):
        return self.solver.modes[n_mode].neff
    
    def get_intensity(self, n_mode=0):
        # e = self.get_Efield(n_mode)
        e = self.formated_data[n_mode][4]
        # return np.real(self.solver.modes[n_mode].intensity())
        return abs(e[0])**2+abs(e[1])**2+abs(e[2])**2
        
    def get_intensity_new(self, n_mode=0):
        # e = self.get_Efield(n_mode)
        return np.real(self.solver.modes[n_mode].intensity())
        # return abs(e[0])**2+abs(e[1])**2+abs(e[2])**2
    
    def get_total_power(self, n_mode=0):
        e = self.get_Efield(n_mode)
        h = self.get_Hfield(n_mode)
        pointing_z = e[0]*np.conj(h[1])-e[1]*np.conj(h[0])
        dx = self.x[1]-self.x[0]
        dy = self.y[1]-self.y[0]
        return 0.5*np.real(pointing_z.sum())*dx*dy   # 0.5 because of complex notation https://fr.wikipedia.org/wiki/Vecteur_de_Poynting
    
    def plot_intensity(self, n_mode=0):
        plt.figure()    
        width = self.width
        height = self.height
        x = self.x
        y = self.y 
        i = self.get_intensity(n_mode).T[0] 
        plt.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                  [-height/2, -height/2, height/2, height/2,-height/2],'white')
        plt.imshow(i,
                   extent=np.array([x.min(),x.max(),y.min(),y.max()]),
                   origin='lower',
                   # aspect=(y.max()-y.min())/(x.max()-x.min()),
                   cmap='jet')
        
    def plot_field(self, n_mode=0):
        # Ex, Ey, Ez = self.get_Efield(n_mode)
        Ex, Ey, Ez = self.formated_data[n_mode][4]
        Ex = Ex[:,:,0]
        Ey = Ey[:,:,0]
        Ez = Ez[:,:,0]
        width = self.width
        height = self.height
        x = self.x
        y = self.y 
        fig = pylab.figure()
        fig.add_subplot(1, 3, 1)
        pylab.contourf(x, y, abs(Ex), 50)
        plt.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                  [-height/2, -height/2, height/2, height/2,-height/2],'r')
        pylab.title('$E_x$')
        fig.add_subplot(1, 3, 2)
        pylab.contourf(x, y, abs(Ey), 50)
        plt.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                  [-height/2, -height/2, height/2, height/2,-height/2],'r')
        pylab.title('$E_y$')
        fig.add_subplot(1, 3, 3)
        pylab.contourf(x, y, abs(Ez), 50)
        plt.plot([-width/2, width/2, width/2, -width/2,-width/2], 
                  [-height/2, -height/2, height/2, height/2,-height/2],'r')
        pylab.title('$E_z$')
        pylab.show()
        
    def get_n_lambda(self, lamda, material):
        lamda = lamda*1e6
        if material=='SiN':
            n = np.sqrt(1+3.0249*lamda**2/(lamda**2 -0.1353406**2) + 40314*lamda**2/(lamda**2 - 1239.842**2))  
        elif material=='SiO2':
            n = np.sqrt(1+0.6961663*lamda**2/(lamda**2 -0.0684043**2) + 0.4079426*lamda**2/(lamda**2 - 0.1162414**2)+ 0.8974794*lamda**2/(lamda**2 - 9.896161**2)) 
        else:
            raise "Material not found"
        return n
    
    def get_formatted_array(self, n_mode):
        return np.array([self.wavelength, self.x, self.y, np.array([0]), self.get_Efield(n_mode=n_mode)], dtype=object)
    
    def was_already_simulated(self, n_mode):
        # check if there is already a simulation for given epsfunction, mode number, wavelength and grid density
        filename = "H_%i_W_%i_%i_eigen%i_%i"%(self.height*1e9, self.width*1e9, self.wavelength*1e10, n_mode+1, self.gridsize)
        npyname = os.path.join(self.datafolder, filename+".npz")
        if os.path.isfile(npyname):
            return True
        return False
            
    def were_already_simulated(self):
        for i in range(self.neigs):
            if not self.was_already_simulated(i):
                return False
        return True
    
    def save_simulation(self, n_mode):
        # I want to save E, wavelength, mode number, epsfunction, grid density, effective refractive index
        # To make a file compatible with nanotrappy requirement, I should use the format [lambda, x, y, z, Efield]
        filename = "H_%i_W_%i_%i_eigen%i_%i"%(self.height*1e9, self.width*1e9, self.wavelength*1e10, n_mode+1, self.gridsize)
        # stuff = {
        #     "formatted_array":self.get_formatted_array(n_mode),
        #     "neff": self.get_neff(n_mode)
        #     }
        # np.save(filename, stuff)
        formatted_array = self.get_formatted_array(n_mode)
        neff = self.get_neff(n_mode)
        os.makedirs(self.datafolder, exist_ok=True)
        np.savez_compressed(os.path.join(self.datafolder, filename), formatted_array=formatted_array, neff=neff)
        
    def save_simulations(self):
        for i in range(self.neigs):
            self.save_simulation(i)
    
    def load_simulation(self, n_mode):
        filename = "H_%i_W_%i_%i_eigen%i_%i"%(self.height*1e9, self.width*1e9, self.wavelength*1e10, n_mode+1, self.gridsize)
        if self.was_already_simulated(n_mode):
            data = np.load(os.path.join(self.datafolder, filename+".npz"), allow_pickle=True)
            return data['formatted_array'], data['neff']
        raise "cannot load simulation"
    
    def is_guided(self, n_mode):
        isguided = self.get_n_lambda(self.wavelength, "SiO2") < self.neff[n_mode]
        s = "Mode %i: n_SiO2 = %.4f, neff = %.4f"%(n_mode, self.get_n_lambda(self.wavelength, "SiO2"), np.real(self.neff[n_mode]))
        if isguided:
            print(s+ ". IS GUIDED.")
        else:
            print(s+ ". IS NOT GUIDED.")
        return isguided
    
    
    def get_line_v(self, n_mode = 0, ncol=None):
        Ex, Ey, Ez = self.formated_data[n_mode][4]
        Ex = Ex[:,:,0]
        Ey = Ey[:,:,0]
        Ez = Ez[:,:,0]
        ncol = int(Ex.shape[0]/2) if ncol is None else ncol
        Ex_col = Ex[:,ncol]
        Ey_col = Ey[:,ncol]
        Ez_col = Ez[:,ncol]
        return Ex_col, Ey_col, Ez_col
    
    def plot_Ez_o_Ex(self, nmode=0, ncol=None):
        ex, ey, ez = self.get_line_v(nmode, ncol)
        plt.plot(self.y, np.imag(ez)/np.real(ey), label='ez/ey, h=%.0f nm'%(self.height*1e9))
        plt.axvline(x= self.height/2, color='black')
        # plt.plot(self.y, abs(ez), label='ez')
        # plt.plot(self.y, abs(ey), label='ey')
        # plt.plot(self.y, abs(ex), label='ex')
        plt.legend()
        # Ex, Ey, Ez = self.formated_data[nmode][4]
        # plt.figure()
        # plt.imshow(abs(Ez[:,20:80])/abs(Ex[:,20:80]))
        

def fictitous_compensation(s1, s2, nmode=0):  # Detuning in nanometers
    b1 = s1.get_fictitious_B(nmode)
    by1 = b1[1,:,:,0]
    x=s1.x
    y=s1.y
    for i in range(len(by1)):
        for j in range(len(by1[0])):
            if abs(x[i])<(s1.width/2) and abs(y[j])<(s1.height/2):
                by1[j,i]=0
    
    b2 = s2.get_fictitious_B(nmode)
    by2 = b2[1,:,:,0]
    x=s2.x
    y=s2.y
    for i in range(len(by2)):
        for j in range(len(by2[0])):
            if abs(x[i])<(s2.width/2) and abs(y[j])<(s2.height/2):
                by2[j,i]=0
    res=(np.real(by1)-np.real(by2))
    # res[abs(res)>0.155]=0
    return res

if __name__=="__main__":
    t=time.time()
    s = simulation(wavelength=690e-9,  gridsize=1000, height=150e-9, nmodes=2)
    tf = time.time()-t
    print(tf)
    # s.plot_fictitious_By(0)
    # s2 = simulation(wavelength=690.0e-9,  gridsize=500, height=150e-9, nmodes=2)
    
    # plt.imshow(fictitous_compensation(s1, s2, 0), cmap='jet', origin='lower')
    # plt.colorbar()
    # plt.plot(s1.x*1e6, np.real(fictitous_compensation(s1, s2, 0)[230]), label="Compensated")
    # plt.plot(s1.x*1e6, np.real(s1.get_fictitious_B(0)[1,:,:,0][230]), label="Non compensated")
    # plt.xlabel("Position along width of waveguide, just above waveguide ($\mu m$)")
    # plt.ylabel("Fictitious B (possibly Gauss)") 
    # plt.title("Atttenuation of the fictitious B field by using \n a counterpropagating blue detune dipole beam.\n \
    #           1st beam @690 nm, 2nd @ 692 nm \n Both polarizations are TE")
    # plt.legend()
    # plt.tight_layout()
    # s.plot_field(0)
    # s1.plot_fictitious_By(0)
    # f=s.get_fictitious_B(0)
    # s1.plot_field(0)
    
    # plt.figure()
    # plt.plot(s1.x*1e6, np.real(fictitous_compensation(s1, s2, 0)[230])/np.real(s1.get_fictitious_B(0)[1,:,:,0][230]))
    
    
    
    
    
    
    
    
    # import arc
    # from arc import Rubidium87
    # from scipy.constants import h
    # from nanotrappy import atomicsystem

    # atom = Rubidium87()
    # pol = arc.calculations_atom_single.DynamicPolarizability(atom, 5, 0, 1/2)
    # pol.defineBasis(5, 16)
    # p = pol.getPolarizability(900e-9, units="SI")
    # print(p[1])

    # atomic_system = atomicsystem(atom,"5S1/2", f=1)
    # atomic_system.set_alphas(900e-9)
    # # s = atomic_system.alpha_scalar(self.wavelength) 
    # v = atomic_system.alpha_vector(900e-9)
    # print(v/h)
    
    
    
    


    # s.plot_Ez_o_Ex(1)
    # a=s.get_fictitious_B(1)
    # print(a)
    # ex, ey, ez = s.get_line_v(1)
    # ex, ey, ez = s.formated_data[1][4]
    # x,y,z = s.formated_data[1][1:4]
    # ex = ex[:,:,0]
    # ey = ey[:,:,0]
    # ez = ez[:,:,0]
    # n = 100
    # plt.clf()
    # # plt.plot(y, np.real(ex[:,n]), label='ex')
    # # # for i in range(230,270):
    # # plt.plot(y, np.real(ey[:,n]), label='ey')
    # # plt.plot(y, np.imag(ez[:,n]), label='ez')
    # # plt.legend()
    # # plt.imshow(np.imag(ez))
    # plt.plot(np.imag(ez[:,n])/np.real(ey[:,n]))
    
    # eyy = ey[135,n]
    # ezz = ez[135,n]
    # beta = abs(ezz)/abs(eyy)
    # proj = 0.5*abs(eyy-1j*ezz)**2/(abs(eyy)**2+abs(ezz)**2)

    # plt.clf()
    # proj = []
    # betas = np.linspace(0.0, 1,100)
    # for beta in betas:
    #     eyy = 1
    #     ezz = 1j*beta
    #     p = 0.5*abs(eyy+1j*ezz)**2/(abs(eyy)**2+abs(ezz)**2)
    #     proj.append(p*(1-p))
    # plt.plot(betas, proj)
    # plt.xlabel("beta")
    # plt.ylabel("projection")
    
    
    # betas = np.linspace(0.0, 1,500)
    # contrasts = []
    # for beta in betas:
    #     from scipy.constants import pi
    #     z = np.linspace(0, 4*780e-9, 500) 
    #     eyyp = eyy*np.exp(1j*2*pi/780e-9*1.5*z)
    #     ezzp = 1j*eyy*beta*np.exp(1j*2*pi/780e-9*1.5*z)
        
    #     # ix = np.abs(ex[line]-np.conj(ex[line]))**2
    #     iy = np.abs(eyyp-np.conj(eyyp))**2
    #     iz = np.abs(ezzp-np.conj(ezzp))**2
    #     ti = iy+iz
    #     ti_max=ti.max()
    #     ti_min = ti.min()
    #     contrast = (ti_max-ti_min)/(ti_max+ti_min)
    #     contrasts.append(contrast)
    #     print(contrast)
    
    # plt.plot(betas, contrasts)
    # plt.xlabel("Amplitude ratio $\|E_z\|/\|E_y\|$" )
    # plt.ylabel("Contrast of the standing wave")
    # plt.title("")
    
    # plt.plot(z, iy, label="$\|E_y\|^2$")
    # plt.plot(z, iz, label="$\|E_z\|^2$")
    # plt.plot(z, ti, label="$\|\overrightarrow{E}\|^2=\|E_y\|^2+\|E_z\|^2$")
    # plt.xlabel("Position along fiber (nm)")
    # plt.ylabel("Intensity (a.u.)")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    
    
    # dens = np.array([int(el) for el in np.linspace(20,500,20)])
    # neffs = []
    # times = []
    # for el in dens:
    #     print(el)
    #     start_time = time.time()
    #     s = simulation(gridsize=el)
    #     nn = s.get_neff()
    #     neffs.append(nn)
    #     print(nn)
    #     times.append(time.time() - start_time)
    # wavelengths = [690e-9, 780e-9, 900e-9]
    # s = [simulation(wavelength=wl,  gridsize=400, height=150e-9, nmodes=2) for wl in wavelengths]
    # for i in np.linspace(260e-9, 300e-9, 1):
    #     s = simulation(wavelength=780e-9,  gridsize=300, height=i, nmodes=2)
    #     s.plot_Ez_o_Ex(nmode=1)
    # s.is_guided(0)
    # s.is_guided(1)
    # s.plot_field(n_mode=0)
    # plt.plot(dens, neffs,"o-") 
    # plt.xlabel("Mesh size")
    # plt.ylabel("n_eff")
    
    # plt.plot(dens, times, "o-")
    # plt.xlabel("Mesh size")
    # plt.ylabel("Time of execution (s)")
    # s = simulation()
    # s.plot_intensity()
    # m = s.solver.modes[0]
    # m.normalize
    # s.plot_intensity(0)
    # s.plot_field()
    # i = s.get_intensity()
    # plt.imshow(i.T, origin='lower')
    # s.plot_field(0)
    # res = s.get_total_power(0)
    # print(res)
    # s.plot_field()
    # plt.figure()
    # s.plot_n()
