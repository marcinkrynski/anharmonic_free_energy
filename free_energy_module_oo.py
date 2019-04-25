import numpy as np
import os


hb_J = 1.0545718e-34 #J*s
kb_J = 1.38064852*10**-23 #J/K
kb_ev = 8.6173303*10**-5 #eV/K
ev2J = 1.60217662*10**-19
J2ev = 1/ev2J
cm2hz = .02998*10**12
atomic2s = 2.418* 10**-17

#returns list of files in fpath directory
def list_of_files(fpath):
    return [f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))]

#returns list of folders in fpath directory
def list_of_directories(fpath):
    return list(set(os.listdir(fpath))-set(list_of_files(fpath)))

#performs integration with trapezoid method
def intgrt(x, y):
    I = [0]
    err = [0]
    for i in range(1, len(x)):
        I.append(np.trapz(y[:i+1], x[:i+1]))
        y_p = np.gradient(y[:i+1])
        err.append((((x[-1]-x[0])**2)/(12*len(x)**2))*(y_p[-1]-y_p[0]))
        
    I = np.array(I)
    err = np.array(err)
    return I, err

def lammps_log_to_U_latt(fpath):
    file = open(fpath+'log.lammps')
    lines = file.readlines()
    file.close()
    last_step = [i-1 for i in range(len(lines)) if lines[i][:9]=='Loop time'][0]
    return float(lines[last_step].split()[1])

#calculates harmonic freee energy - quantum and classic - at given temperature at gamma
#reads eigenvalues in atomic units, omiting first line - ipi format
def ipi_harmonic_free_energy(fpath, T, nmols, U_latt):
    file = open(fpath)
    lines = file.readlines()[1:]
    file.close()

    omega_kw = np.array(lines, float)[3:]
    omega_kw_abs = np.abs(omega_kw)
    omega = omega_kw_abs**.5

    freq = omega/(2*np.pi)
    freq =  freq  / atomic2s
    omega = freq * 2*np.pi
    
    Fq = []
    Fc = []
    for T_ in T:        
        beta = 1/(T_*kb_J)
        Fq.append(
                np.sum(
                        hb_J*omega/2 + (beta**-1)*np.log(1-np.exp(-hb_J*omega*beta))
                        )
                )
        Fc.append(
                np.sum(
                        np.log(beta*hb_J*omega)/beta
                        )
                )
    return np.asanyarray(Fq)*J2ev/nmols + U_latt, np.asanyarray(Fc)*J2ev/nmols + U_latt

#calculates harmonic freee energy - quantum and classic - at given temperature
#reads eigenvalues in atomic units, omiting first line - ipi format
def phonopy_harmonic_free_energy(fpath, T, nmols, U_latt):
    file = open(fpath)
    lines = file.readlines()[3:]
    file.close()

    data = np.array([l.split() for l in lines], float)
    f = data[:,0]*cm2hz
    omega = f * 2*np.pi
    dos = data[:,1]
    
    Fq = []
    Fc = []
    for T_ in T:        
        beta = 1/(T_*kb_J)
        Fq.append(
                np.sum(
                        dos*(
                                hb_J*omega/2 + (beta**-1)*np.log(1-np.exp(-hb_J*omega*beta))
                                )
                        )
                )
        Fc.append(
                np.sum(
                        dos*(
                                np.log(beta*hb_J*omega)/beta
                                )
                        )
                )
    return np.asanyarray(Fq)*J2ev/nmols + U_latt, np.asanyarray(Fc)*J2ev/nmols + U_latt

def _error_from_u(u):
    ug = np.gradient(u)
    c = []
    for i in range(1000):
        t1 = np.arange(0, len(ug)-i,1)
        t2 = np.arange(i, len(ug),1)
        c.append(np.sum(ug[t1]*ug[t2]))
    c = np.array(c)
    c = c/c[0]
    c = c*np.sign(c)
    c_trs = np.sum(c>.1)
    N_independent = len(u)/c_trs
    return np.std(u)/(N_independent**.5)

#reads md temperature and potential energy from simulation.out files
#within subfolders in folder 'fpath'
#steps_excluded is number of initial excluded steps
#returns temperature in K and potential energy in eV
#both quantities are sorted with respect to the temperature
def ipi_md_potential(fpath, nmols, bexclude=1000):
    subfolders = np.array(list_of_directories(fpath))
    sub = np.array(subfolders, int)
    idx = np.argsort(sub)
    subfolders = subfolders[idx]
        
    T = []
    U = []
    err = []
    for folder in subfolders:
        fpath_ = fpath+folder
        file = open(fpath_+'/simulation.out')
        lines = file.readlines()
        file.close()
        head = [l for l in lines[:20] if l[0]=='#']
        lines = lines[len(head):]
        head = [h.split('-->')[1] for h in head]
        head = [h.split(' : ')[0] for h in head]
        data_matrix = np.array([l.split() for l in lines], float)[bexclude:,:]

        T.append(np.mean(data_matrix[:,3]))
        U.append(np.mean(data_matrix[:,5]))
        err.append(
                _error_from_u(data_matrix[:,5])
                )
    return np.asanyarray(T), np.asanyarray(subfolders, float), np.asanyarray(U)/nmols, np.asanyarray(err)/nmols

#returns harmonic potential energy for given nuner of atoms
def harmonic_potential_energy(T,N):
    return J2ev*(3*N-3)*kb_J*T/2

#reruns anahrmonic energy
def anharmonic_energy(U_md, U_harm, U_latt):
    return U_md - U_harm - U_latt

#returns integrated anharmonic energy
def integrated_anharmonic_energy(T, U_anharm, U_err):
    xs = T
    ys = U_anharm/(T**2)/kb_ev
    ys_err = U_err/(T**2)/kb_ev
    
    xl = np.log(T/T[0])
    yl = U_anharm/np.exp(xl)/T[0]/kb_ev
    yl_err = U_err/np.exp(xl)/T[0]/kb_ev

    int_s = intgrt(xs, ys)
    int_l = intgrt(xl, yl)
    
    int_s_err = intgrt(xs, ys_err)
    int_l_err = intgrt(xl, yl_err)

    return int_s[0], int_s[1], int_l[0], int_l[1], int_s_err[0], int_l_err[0]

#returns temperature and potential energy ffrom FF->DFT calculation
def ipi_to_two_potentials(path, cut):
    file = open(path)
    lines = file.readlines()[1:]
    file.close()
    
    idx = [i for i in range(len(lines)) if lines[i][0]=='#'][-1]+1
    lines = lines[idx:]

    data = np.array([l.split() for l in lines], float)
    time = data[cut:, 1]
    pot_1 = data[cut:, 7]
    pot_2 = data[cut:, 8]
    
    return time, pot_1, pot_2

class fesample:
    def __init__(
            self,
            ulatt_fpath, ulatt_nmols,
            md_fpath, md_nmols,
            u_harm_natoms, u_harm_nmols,
            fharm_fpath_phonopy, fharm_nmols_phonopy,
            fharm_fpath_ipi, fharm_nmols_ipi,
            ):
        self.__ulatt_fpath = ulatt_fpath
        self.__ulatt_nmols = ulatt_nmols
        self.__md_fpath = md_fpath
        self.__md_nmols = md_nmols
        self.__u_harm_natoms = u_harm_natoms
        self.__u_harm_nmols = u_harm_nmols
        self.__fharm_fpath_phonopy = fharm_fpath_phonopy
        self.__fharm_nmols_phonopy = fharm_nmols_phonopy
        self.__fharm_fpath_ipi = fharm_fpath_ipi
        self.__fharm_nmols_ipi = fharm_nmols_ipi
        
        #read lattice energy
        self.U_latt = lammps_log_to_U_latt(self.__ulatt_fpath )/self.__ulatt_nmols
        
        #read temperature and u_md
        self.T, self.Tf, self.u_md, self.u_md_err = ipi_md_potential(
                self.__md_fpath,
                self.__md_nmols,
                bexclude=1000,
                )
        
        #calculate harmonic free energy, quantum and classical
        self.F_harm_q_ipi, self.F_harm_c_ipi = ipi_harmonic_free_energy(
                self.__fharm_fpath_ipi,
                self.Tf,
                self.__fharm_nmols_ipi,
                self.U_latt,
                )
        self.F_harm_q_phonopy, self.F_harm_c_phonopy = phonopy_harmonic_free_energy(
                self.__fharm_fpath_phonopy,
                self.Tf,
                self.__fharm_nmols_phonopy,
                self.U_latt,
                )


        ###calculatin angarmonic free energy##
        ###calculatin angarmonic free energy##
        #calculating anharmonic integral
        self.u_harm = harmonic_potential_energy(self.Tf, self.__u_harm_natoms)/self.__u_harm_nmols
        self.u_anharm = anharmonic_energy(self.u_md, self.u_harm, self.U_latt)
        (
            self.anharm_integral_str,
            self.anharm_integral_str_err,
            self.anharm_integral_log,
            self.anharm_integral_log_err,
            self.anharm_integral_str_md_err,
            self.anharm_integral_log_md_err,
        ) = integrated_anharmonic_energy(self.Tf, self.u_anharm, self.u_md_err)

        #calculating harmonic part
        self.f_harm_part = (self.F_harm_c_ipi[0]-self.U_latt)*self.Tf/self.Tf[0]
        
        #calculating kinetic part
        self.f_class_nucl = kb_ev*self.Tf*(3*self.__u_harm_natoms-3)*np.log(self.Tf/self.Tf[0])/self.__u_harm_nmols
        
        #calculating classical anharmonic free energy
        self.F_anh_c = self.U_latt + self.f_harm_part - self.f_class_nucl - self.anharm_integral_log*kb_ev*self.Tf
        
        #calculating quantum anharmonic free energy
        self.F_anh_q = self.F_anh_c + self.F_harm_q_ipi - self.F_harm_c_ipi
        
        #calculating the error of the TI parht of anharmonic free energy
        self.F_err_int_ti_log = self.Tf*self.anharm_integral_log_err*kb_ev
        self.F_err_int_ti_str = self.Tf*self.anharm_integral_str_err*kb_ev
        
        #calculating the error of the TI parht of anharmonic free energy
        self.F_err_md_str = self.Tf*self.anharm_integral_str_md_err*kb_ev
        self.F_err_md_log = self.Tf*self.anharm_integral_log_md_err*kb_ev
        
        self.F_anh_err = (
                np.absolute(self.F_err_md_log) +
                np.absolute(self.F_err_int_ti_log)
                )
