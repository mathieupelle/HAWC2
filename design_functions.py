# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 10:54:40 2021

@author: Mathieu Pellé
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize  import  least_squares
from scipy.interpolate import CubicSpline
import shutil
import os
import re

class HAWC_design_functions:
    def __init__(self):
        print('>> Aero design class initialised...')


    def Radius_Scaling(self, R_ref, V_rated_ref, turb_class_ref, turb_class_target):
        print('>> Scaling for new wind class...')

        turb_classes = ['A', 'B', 'C']
        turb_intensities = [0.16, 0.14, 0.12]
        I_ref = turb_intensities[turb_classes.index(turb_class_ref)]
        self.I = turb_intensities[turb_classes.index(turb_class_target)]


        V_ref_max = V_rated_ref*(1+2*I_ref)

        R2_guess = R_ref*1.1
        dif = 1e12;
        it = 0;
        while dif>1e-6:

            self.V_rated = (V_rated_ref**3*R_ref**2/R2_guess**2)**(1/3)
            V_new_max = self.V_rated*(1+2*self.I)

            self.R = R_ref*V_ref_max/V_new_max
            dif = abs(self.R - R2_guess)
            R2_guess = self.R
            it = it + 1

        print('   New rotor radius: '+str(round(self.R,3))+' m')


    def Airfoil_Tuning(self, cl_shift, remove = False, polars = True):
        print('>> Selecting airfoil design points...')

        aero_data_path = './Aerodynamics/Airfoil data/'

        filenames = [f for f in listdir(aero_data_path) if isfile(join(aero_data_path, f))]


        if type(remove) == list:
            for i in range(len(remove)):
                filenames.remove(remove[i])

        self.cl_des, self.cd_des, self.alpha_des, self.tcratio_af = np.zeros((4,1,len(filenames)))

        for i in range(len(filenames)):

            self.tcratio_af[0 ,i] = float(filenames[i][7:10])/10

            data = np.loadtxt(aero_data_path + filenames[i])
            data = data[np.argmin(abs(data[:, 0]+15)):np.argmin(abs(data[:, 0]-30)), :]

            idx_min = np.argmin(abs(data[:, 0]+5))
            idx_max = np.argmin(abs(data[:, 0]-20))
            cl_max = max(data[idx_min:idx_max, 1])
            idx = np.where(data == cl_max)
            idx = idx[0][0]

            self.cl_des[0, i] = cl_max - cl_shift[i]

            self.alpha_des[0, i] = np.interp(self.cl_des[0, i], data[idx_min:idx, 1], data[idx_min:idx, 0])
            self.cd_des[0, i] = np.interp(self.cl_des[0, i], data[idx_min:idx, 1], data[idx_min:idx, 2])


            if polars:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.axhline(y = self.cl_des[0, i], color='r', linestyle='--', alpha=0.5)
                ax1.plot(data[:, 0], data[:, 1], '-.k')
                ax1.plot(self.alpha_des[0, i], self.cl_des[0, i], 'xr')
                ax1.set(xlabel = r'$\alpha$ [deg]', ylabel = '$C_l$ [-]')
                ax1.grid()
                ax1.set_xlim(min(data[:, 0]), max(data[:, 0]))

                ax2.plot(data[:, 2], data[:, 1], '-.k')
                ax2.axhline(y = self.cl_des[0, i], color='r', linestyle='--', alpha=0.5)
                ax2.axvline(x = self.cd_des[0, i], color='r', linestyle='--', alpha=0.5)
                ax2.plot(self.cd_des[0, i], self.cl_des[0, i], 'xr')
                ax2.set(xlabel = '$C_d$ [-]')
                ax2.grid()
                plt.suptitle(filenames[i][0:-4])

        self.cl_des = np.append(self.cl_des, np.array([[0]]))
        self.alpha_des = np.append(self.alpha_des, np.array([[0]]))
        self.cd_des = np.append(self.cd_des, np.array([[0.6]]))
        self.tcratio_af = np.append(self.tcratio_af, np.array([[100]]))
        self.clcd_des = self.cl_des/self.cd_des



    def Fit_Polynomials(self, order, R_ref, plotting=True):
        print('>> Fitting polynomials...')

        y = [self.cl_des, self.clcd_des, self.alpha_des]
        names = ['cl', 'clcd', 'alpha', 'thickness']
        self.coefs = dict.fromkeys(names, None)
        lab = ['Design $C_l$ [-]', 'Design $C_l/C_d$ [-]', r'Design $\alpha$ [deg]']
        for i in range(3):
            z1 = np.polyfit(self.tcratio_af[0:4], y[i][0:4], order[i])
            z2 = np.polyfit(self.tcratio_af[3:], y[i][3:], 1)
            self.coefs[names[i]] = [z1, z2]
            p1 = np.poly1d(z1)
            p2 = np.poly1d(z2)
            x = np.linspace(20, 100, 1000)

            if plotting:
                plt.figure()
                plt.plot(x[x<=self.tcratio_af[3]], p1(x[x<=self.tcratio_af[3]]), 'b')
                plt.plot(x[x>=self.tcratio_af[3]], p2(x[x>=self.tcratio_af[3]]), 'b')
                plt.plot(self.tcratio_af, y[i], '+k')
                plt.grid()
                plt.xlabel('$t/c$ [\%]')
                plt.ylabel(lab[i])

        ae_path = './DTU10MW/data/DTU_10MW_RWT_ae.dat'
        data = np.genfromtxt(ae_path, skip_header=2, delimiter='\t')


        t_ref = data[:,2]*data[:,1]/100
        self.t_max = max(t_ref)
        t = t_ref*self.R/R_ref
        r = np.linspace(0, self.R , len(t_ref))

        z = np.polyfit(r, t, order[-1])
        self.coefs[names[-1]] = z
        p = np.poly1d(z)
        y = p(np.linspace(0, R_ref, 100))

        if plotting:
            plt.figure()
            plt.plot(data[:,0]/R_ref, t_ref, 'x', label='DTU 10MW')
            plt.plot(r/self.R, t, 'x', label='New design')
            plt.plot(np.linspace(0, 1, 100), y, '--k',  label='Fit')
            plt.xlabel('r/R [-]')
            plt.ylabel('t [m]')
            plt.grid()
            plt.legend()


    def Thickness_poly(self, r):

        if r <= 5:
            r = 5
        p = np.poly1d(self.coefs['thickness'])
        t = p(r)

        return t


    def Alpha_poly(self, tcratio):

        if tcratio<24:
            tcratio = 24
        elif tcratio>100:
            tcratio = 100

        if tcratio < 48:
            p = np.poly1d(self.coefs['alpha'][0])
            alpha = p(tcratio)
        elif 36 <= tcratio:
            p = np.poly1d(self.coefs['alpha'][1])
            alpha = p(tcratio)

        return alpha


    def Cl_poly(self, tcratio):

        if tcratio<24:
            tcratio = 24
        elif tcratio>100:
            tcratio = 100

        if tcratio < 48:
            p = np.poly1d(self.coefs['cl'][0])
            cl = p(tcratio)
        elif 48 <= tcratio:
            p = np.poly1d(self.coefs['cl'][1])
            cl = p(tcratio)

        return cl

    def Clcd_poly(self, tcratio):

        if tcratio<24:
            tcratio = 24
        elif tcratio>100:
            tcratio = 100

        if tcratio < 48:
            p = np.poly1d(self.coefs['clcd'][0])
            clcd = p(tcratio)
        elif 48 <= tcratio:
            p = np.poly1d(self.coefs['clcd'][1])
            clcd = p(tcratio)

        return clcd

    def Residuals(self, x):

        c = x[0]
        ap = x[1]
        a = 1/3

        t = self.Thickness_poly(self.r)
        if t > self.t_max:
            t = self.t_max

        tcratio = t/c*100;

        clcd = self.Clcd_poly(tcratio)
        cl = self.Cl_poly(tcratio)
        phi = np.arctan((1-a)*self.R/((1+ap)*self.r*self.TSR));
        cd = cl/clcd;
        cy = cl*np.cos(phi) + cd*np.sin(phi);
        cx = cl*np.sin(phi) - cd*np.cos(phi);
        f = self.B/2*(self.R-self.r)/(self.r*np.sin(phi));
        F = 2/np.pi*np.arccos(np.exp(-f));
        sigma = self.B*c/(2*np.pi*self.r);


        res_c = 4*np.pi*self.r*np.sin(phi)**2*F*2*a/(cy*self.B*(1-a)) - c;
        res_ap = 1/(4*F*np.sin(phi)*np.cos(phi)/sigma/cx-1) - ap;
        res = [res_c, res_ap];

        idx = np.where(self.r_lst == self.r)
        self.t[idx] = t
        self.c[idx] = c
        self.tcratio[idx] = tcratio
        self.phi[idx] = phi
        self.alpha[idx] = self.Alpha_poly(tcratio)
        self.beta[idx] = np.rad2deg(phi)-self.Alpha_poly(tcratio)
        self.cl[idx] = cl
        self.cd[idx] = cd
        self.ap[idx] = ap
        self.cp[idx] = ((1-a)**2 + (self.TSR*self.r/self.R)**2*(1+ap)**2)*self.TSR*self.r/self.R*sigma*cx
        self.ct[idx] = ((1-a)**2 + (self.TSR*self.r/self.R)**2*(1+ap)**2)*sigma*cy
        self.a[idx] = np.round(1/(4*F*np.sin(phi)**2/(sigma*cy)+1),4)

        return np.array(res)

    def Chord_Optimisation(self, B=3, TSR=8.0, N=200, plotting=True):
        print('>> Optimising chord...')

        self.TSR = TSR
        self.B = B

        self.r_lst = np.linspace(5, self.R*0.95, N)
        self.t, self.c, self.tcratio, self.phi, self.alpha, self.beta, self.cl, self.cd, self.ap, self.a, self.cp, self.ct, self.CP, self.CT = np.zeros((14, N, 1))

        x0 = np.array([6.0, 0.001])
        bounds = ((0, 0), (np.inf, 1))
        out = np.empty((self.r_lst.size, 2))

        for i in range(len(self.r_lst)):
            self.r = self.r_lst[i]
            res = least_squares(self.Residuals, x0, bounds=bounds)
            out[i] = res.x
            x0 = res.x

        self.CP = np.trapz(np.multiply(self.cp.T, self.r_lst), self.r_lst)*2/self.R**2
        self.CT = np.trapz(np.multiply(self.ct.T, self.r_lst), self.r_lst)*2/self.R**2

        var_lst = [self.c, self.alpha, self.cl, self.cl/self.cd, self.a, self.ap, self.beta, self.cp, self.ct]
        labels = ['$c$ [m]', r'$\alpha$ [deg]', '$C_l$ [-]', '$C_l/C_d$ [-]', '$a$ [-]', "$a'$ [-]", r'$\beta$ [deg]', '$C_p$ [-]', '$C_t$ [-]']

        if plotting:
            for j, var in enumerate(var_lst):
                plt.figure()
                plt.plot(self.r_lst, var, '-b')
                plt.xlabel('r [m]')
                plt.ylabel(labels[j])
                plt.grid()

                if j == 4:
                    plt.axhline(y=1/3, color='k', linestyle='--', alpha=0.5)
                elif j == 7:
                    plt.axhline(y=16/27, color='k', linestyle='--', alpha=0.5)
                elif j ==8:
                    plt.axhline(y=8/9, color='k', linestyle='--', alpha=0.5)
        print('>> CP: '+ str(round(self.CP[0],3)) + ', CT:' + str(round(self.CT[0],3)))

    def Limits_and_Smoothing(self, R_ref, plotting=True):
        print('>> Tuning blade design...')

        ae_path = './DTU10MW/data/bladedat.txt'
        data = np.genfromtxt(ae_path, delimiter='\t')
        r_ref = data[:,0]
        c_ref = data[:,2]

        a = r_ref[0:6]
        b = c_ref[0:6]
        b[4] = b[4]*1.02
        b[5] = b[5]*1.03

        for i in range(len(self.r_lst)):
            if self.beta[i] > 25:
                self.beta[i] = 25

            tcratio = self.tcratio[i]
            if tcratio < 24.1:
                tcratio = 24.1
                self.c[i] = self.t[i]/(tcratio/100)

            cs = CubicSpline(a, b)

            if self.r_lst[i] < 43 and self.r_lst[i] > 9.2:
                self.c[i] = cs(self.r_lst[i])*self.R/R_ref

            elif self.r_lst[i] < 10:
                self.c[i] = np.interp(self.r_lst[i], [self.r_lst[0], 10], [c_ref[0], cs(10)*self.R/R_ref])

            if self.c[i] > self.R/R_ref*max(c_ref):
                self.c[i] = self.R/R_ref*max(c_ref)



        self.r_lst = np.append(self.r_lst, np.array([self.R]))
        self.beta = np.append(self.beta, np.array([self.beta[-1]]))
        self.t = np.append(self.t, np.array([self.t[-1]*0.55]))
        self.c = np.append(self.c, np.array([self.c[-1]*0.55]))

        self.tcratio = self.t/self.c*100


        if plotting:
            labels = [r'$\beta$ [deg]', '$c$ [m]', '$t/c$ [%]', '$t$ [m]']
            data2 = [self.r_lst, self.beta, self.c, self.tcratio, self.t]
            for i in range(len(labels)):
                plt.figure()
                plt.grid()
                plt.xlabel('$r/R$ [-]')
                plt.ylabel(labels[i])
                if i == 3:
                    plt.plot(data[:,0]/R_ref, data[:,3]*data[:,2]/100, '--k', label='DTU 10MW')
                else:
                    plt.plot(data[:,0]/R_ref, data[:,i+1], '--k', label='DTU 10MW')

                plt.plot(data2[0]/self.R, data2[i+1], '-b', label='New design')
                plt.legend()


    def Make_ae_file(self, name):
        print('>> Making ae file...')
        self.new_turbine_name = name
        self.oldpath = './DTU10MW'

        foldernames = [name for name in os.listdir(".") if os.path.isdir(name)]
        if self.new_turbine_name not in foldernames:
            self.newpath = './'+self.new_turbine_name
            shutil.copytree(self.oldpath, self.newpath)
        else:
            print('  Name already exists. Overwriting files...')
            self.newpath = './'+self.new_turbine_name

        newpath_data = self.newpath+'/data'
        filenames = [f for f in listdir(newpath_data) if isfile(join(newpath_data, f))]

        for i in range(len(filenames)):
            prefix = filenames[i][0:12]
            if prefix == 'DTU_10MW_RWT':
                os.rename(newpath_data+ '/' + filenames[i], newpath_data+ '/' +self.new_turbine_name+filenames[i][12:])

        newpath_ae = self.newpath +'/data/'+self.new_turbine_name+'_ae.dat'
        with open(newpath_ae, 'r') as file:
            contents = file.readlines()

        N = len(self.r_lst) + 3
        ae_arr = np.ones((N, 4))
        ae_arr[0,:] = np.array([1, None, None, None])
        ae_arr[1,:] = np.array([1, N-2, None, None])
        ae_arr[2,:] = np.array([0, self.c[0], self.tcratio[0], 1])
        ae_arr[3:,0] = np.array(self.r_lst)
        ae_arr[3:,1] = np.array(self.c)
        ae_arr[3:,2] = np.array(self.tcratio)


        np.savetxt(newpath_ae, ae_arr, delimiter='\t', fmt=' %s')

        with open(newpath_ae, 'r') as file:
            contents = file.readlines()

            for i in range(len(contents)):
                if i<2:
                    contents[i] = contents[i].replace('nan', '')
                    contents[i] = contents[i][1:]
                else:
                    contents[i] = contents[i][1:-1]+'\t ;'+contents[i][-1]

        with open(newpath_ae, 'w') as file:
            file.writelines(contents)

    def Make_htc_steady(self, omega_rated, losses, R_ref, tsr=0, omega_min=0):

        print('>> Making htc file...')

        if tsr == 0:
            tsr = self.TSR

        self.omega_min = omega_min
        self.omega_rated = omega_rated
        self.losses = losses
        self.max_power = 10000*(1+self.losses)


        filenames = [f for f in listdir(self.newpath) if isfile(join(self.newpath, f))]
        for i in range(len(filenames)):
            prefix = filenames[i][0:12]
            if prefix == 'DTU_10MW_RWT':
                os.rename(self.newpath+ '/' + filenames[i], self.newpath+ '/' + self.new_turbine_name+filenames[i][12:])

        filenames = [f for f in listdir(self.newpath) if isfile(join(self.newpath, f))]
        self.newpath_htc_steady = self.newpath+'/'+filenames[1]
        self.newpath_htc_unsteady = self.newpath+'/'+filenames[0]

        ratio = (self.R - 2.8)/(R_ref - 2.8)
        with open(self.oldpath+'/DTU_10MW_RWT_hs2.htc', 'r') as file:
            contents = file.readlines()
            for l, line in enumerate(contents):
                if line.lstrip().startswith('name        blade1'):
                    idx = l+11
            c2_block = contents[idx:idx+27]
            self.c2_block_original = np.zeros((27, 5))
            self.c2_block_new = np.zeros((27, 5))
            for i in range(len(c2_block)):
                lst = re.split(r'\t+', c2_block[i])
                self.c2_block_original[i,:] = lst[2:7]
                lst[3] = str(float(lst[3])*ratio)
                lst[4] = str(float(lst[4])*ratio)
                lst[5] = str(float(lst[5])*ratio)
                twist = -np.interp(float(lst[5]), self.r_lst, self.beta)
                lst[6] = str(twist)
                contents[idx+i] = '\t'.join(map(str,lst))

                self.c2_block_new[i,:] = lst[2:7]


            for i, line in enumerate(contents):

                if line.lstrip().startswith('genspeed'):
                    contents[i] = ('    genspeed '+str(self.omega_min)+' '+str(self.omega_rated)+' ; [rpm]\n')
                if line.lstrip().startswith('opt_lambda'):
                    contents[i] = ('    opt_lambda '+str(tsr)+' ; [-]\n')
                if line.lstrip().startswith('maxpow'):
                    contents[i] = ('    maxpow '+str(self.max_power)+' ; [kW]\n')
                if line.lstrip().startswith('filename'):
                    pos = contents[i].find('DTU_10MW_RWT')
                    contents[i] = contents[i].replace(contents[i][pos:pos+12], self.new_turbine_name)
                if line.lstrip().startswith('ae_filename'):
                    pos = contents[i].find('DTU_10MW_RWT')
                    contents[i] = contents[i].replace(contents[i][pos:pos+12], self.new_turbine_name)
                if line.lstrip().startswith('pc_filename'):
                    pos = contents[i].find('DTU_10MW_RWT')
                    contents[i] = contents[i].replace(contents[i][pos:pos+12], self.new_turbine_name)
        with open(self.newpath_htc_steady, 'w') as file:
            file.writelines(contents)




    def Define_htc_steady_mode(self, mode, blade_distributions=False, control_lst = 0, properties = False):


        print('>> Modifying htc file...')

        freq1 = control_lst[0]
        damp1 = control_lst[1]
        freq2 = control_lst[2]
        damp2 = control_lst[3]
        gain_scheduling = control_lst[4]
        control_type = control_lst[5]

        opt_path = './data/'+str(self.new_turbine_name)+'.opt'
        with open(self.newpath_htc_steady, 'r') as file:
            contents = file.readlines()
            for i, line in enumerate(contents):

                if mode == 'generate_opt' :
                    if line.lstrip().startswith('operational_data_filename'):
                        contents[i] = ('  ;'+'operational_data_filename\t'+str(opt_path)+' ;file with operational data points\n')
                    if line.lstrip().startswith('compute_controller_input'):
                        contents[i] = ('  ;compute_controller_input;\n')
                    if line.lstrip().startswith(';compute_optimal_pitch_angle use_operational_data'):
                        contents[i] = ('  compute_optimal_pitch_angle use_operational_data;\n')

                elif mode == 'controller_tuning':
                    if line.lstrip().startswith(';compute_controller_input'):
                        contents[i] = ('  compute_controller_input;\n')
                    if line.lstrip().startswith(';operational_data_filename'):
                        contents[i] = ('  operational_data_filename\t'+str(opt_path)+' ;file with operational data points\n')
                    if line.lstrip().startswith('compute_optimal_pitch_angle use_operational_data'):
                        contents[i] = ('  ;compute_optimal_pitch_angle use_operational_data;\n')

                    if isinstance(control_lst, list):

                        if line.lstrip().startswith('partial_load'):
                            contents[i] = ('    partial_load '+str(freq1)+' '+str(damp1)+' ; fn [hz], zeta [-]\n')

                        if line.lstrip().startswith('full_load'):
                            contents[i] = ('    full_load '+str(freq2)+' '+str(damp2)+' ; fn [hz], zeta [-]\n')

                        if line.lstrip().startswith('gain_scheduling'):
                            contents[i] = ('    gain_scheduling '+str(gain_scheduling)+' ; 1 linear, 2 quadratic\n')

                        if line.lstrip().startswith('onstant_power'):
                            contents[i] = ('    constant_power '+str(control_type)+'; 0 constant torque, 1 constant power at full load\t\n')

                    else:
                        print('  No controller settings given. Using default...')
                else:
                    print('  Invalid htc mode')


                if blade_distributions:
                    if line.lstrip().startswith(';save_induction'):
                        contents[i] = ('  save_induction;\n')
                else:
                    if line.lstrip().startswith('save_induction'):
                        contents[i] = ('  ;save_induction;\n')
                if properties:
                    if line.lstrip().startswith(';body_output_file_name'):
                        contents[i] = ('body_output_file_name ./info/body.dat;\n')
                    if line.lstrip().startswith(';beam_output_file_name'):
                        contents[i] = ('beam_output_file_name ./info/beam.dat;\n')
                    if line.lstrip().startswith('nbodies     10'):
                        contents[i] = ('    nbodies     1')

        with open(self.newpath_htc_steady, 'w') as file:
            file.writelines(contents)

    def Make_st_file(self):

        newpath_st = './'+ self.new_turbine_name +'/data/'+ self.new_turbine_name +'_Blade_st.dat'
        oldpath_st1 = './DTU10MW/structures/st_original_flexible.dat'
        oldpath_st2 = './DTU10MW/structures/st_original_stiff.dat'

        paths = [oldpath_st1, oldpath_st2]

        s_f_0 = [8,9,13,14,16]
        s_f_1 = [0,2,3,4,5,6,7,17,18]
        s_f_2 = [1,15]
        s_f_4 = [10,11,12]

        files = []
        for f_idx in range(2):

            st_original = np.loadtxt(paths[f_idx], skiprows=5)
            with open(paths[f_idx]) as f:
                a = f.readlines()
            col_name = a[3].split()
            #
            c2_original = self.c2_block_original
            curve_original = np.cumsum ( np.linalg.norm (np.diff (c2_original[:,1:4],
                                                                  axis=0), axis=1))
            #
            c2_new = self.c2_block_new
            curve_new = np.cumsum ( np.linalg.norm (np.diff (c2_new[:,1:4],
                                                                  axis=0), axis=1))
            #
            s_r = curve_new[-1]/curve_original[-1]
            #
            st_new = np.zeros_like(st_original)
            #
            list_0 = [col_name[i] for i in s_f_0]
            list_1 = [col_name[i] for i in s_f_1]
            list_2 = [col_name[i] for i in s_f_2]
            list_4 = [col_name[i] for i in s_f_4]
            #
            s_r = (97.77-2.8)/(178.3/2-2.8)
            #
            for i in s_f_0:
                st_new[:,i] = st_original[:,i] * s_r ** 0.0
            #
            for i in s_f_1:
                st_new[:,i] = st_original[:,i] * s_r ** 1.0
            #
            for i in s_f_2:
                st_new[:,i] = st_original[:,i] * s_r ** 2.0
            #
            for i in s_f_4:
                st_new[:,i] = st_original[:,i] * s_r ** 4.0
            # ============================================================================
            #
            b = []
            if f_idx == 0:
                for i in range(3):
                    b += a[i]
            b += ('    {:7s}    '*19).format(*(col_name))
            b += '\n'
            b += a[4]
            for i, i_t in enumerate(st_new):
                b += ('{:15.6e}').format((st_original[i,0]*s_r))
                b += ('{:15.6e}'*18).format(*(i_t[1:]))
                b+= '\n'
            b = ''.join(b)
            files.append(b)
        fp = open(newpath_st,"w")
        fp.writelines(files[0]+files[1])
        fp.close()


