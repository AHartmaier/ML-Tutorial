import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pylabfea as FE

class Composite(object):
    def __init__(self, param):
        # set elastic material parameters
        self.E1 = param['E1']  # Young's modulus of matrix in GPa
        self.nu1 = param['nu1']  # Poisson's ratio of matrix
        self.E2 = param['E2']  # Young's modulus of filler phase in GPa
        self.nu2 = param['nu2']  # Poisson's ratio of filler phase
        # Check of input values
        if self.E1 < 1 or self.E1 > 600:
            print(f"WARNING: Young's modulus E1 must be in range 1 ... 600 GPa, not {E1}. Using E1=10 GPa.")
            self.E1 = 10.0
        if self.E2 < 1 or self.E2 > 600:
            print(f"WARNING: Young's modulus E2 must be in range 1 ... 600 GPa, not {E2}. Using E2=300 GPa.")
            self.E2 = 300.0
        if self.nu1 < 0.2 or self.nu1 > 0.5:
            print(f"WARNING: Poisson's ratio nu1 must be in range 0.2 ... 0.5, not {nu1}. Using nu1=0.27.")
            self.nu1 = 0.27
        if self.nu2 < 0.2 or self.nu2 > 0.5:
            print(f"WARNING: Poisson's ratio nu2 must be in range 0.2 ... 0.5, not {nu2}. Using nu2=0.27.")
            self.nu2 = 0.27
        
        # set number of finite elements in regular mesh
        self.NX = param['nel']
        self.NY = param['nel']
        # define boundary conditions
        self.sides = 'force' if param['latbc'] == 'free' else 'disp'  # free or fixed lateral BC
        """ Convert strain units from percent to dimensionless """
        self.eps_tot = param['etot']*0.01  # total strain in y-direction
        
        # define phases on composite materials
        self.mat1 = FE.Material(num=1, name="matrix")  # call class to generate material object
        self.mat1.elasticity(E=self.E1*1000, nu=self.nu1)  # define elastic properties
        self.mat2 = FE.Material(num=2, name="filler")  # define second material
        self.mat2.elasticity(E=self.E2*1000, nu=self.nu2)  # material is purely elastic# setup model for elongation in y-direction

        # calculate stress-strain curves of individual phases
        self.mat1.calc_properties(eps=self.eps_tot, sigeps=True)  # invoke method to calculate properties of material 
        self.mat2.calc_properties(eps=self.eps_tot, sigeps=True)  # and to store stress-strain data up to total strain eps=1%

        # initialize variables for FE model
        self.el = None
        self.fe = None
        
        # handels for plots
        self.p_set = set()

    def create_model(self):
        # predefine model with only matrix material
        self.el = np.ones((self.NX, self.NY),dtype=int)
        
        # create and plot model
        self.fe = FE.Model(dim=2, planestress=True)  # initialize finite element model
        self.fe.geom(sect=2, LX=10., LY=10.)  # define geometry with two sections
        self.fe.assign([self.mat1, self.mat2])  # assign materials to sections
        self.fe.mesh(elmts=self.el, NX=self.NX, NY=self.NY)  # create regular mesh with sections as defined in el

    def tensile(self, axis):
        self.fe.u = None  # erase old solutions
        if axis.lower() in ['horizontal', 'horiz', 'hor', 'h']:
            # boundary conditions: uniaxial stress in horizontal direction
            self.fe.bcleft(0.)  # fix left boundary
            self.fe.bctop(0., self.sides)  # boundary condition on lateral edges of model
            self.fe.bcbot(0., self.sides)
            self.fe.bcright(self.eps_tot * self.fe.lenx, 'disp')  # strain applied to rhs nodes
            if self.sides == 'force':
                # fix lateral displacements of corner node to prevent rigid body motion
                hh = [no in self.fe.nobot for no in self.fe.noleft]
                noc = np.nonzero(hh)[0]  # find corner node
                self.fe.bcnode(noc, 0., 'disp', 'y')  # fix lateral displacement
            self.fe.solve()  # calculate distortions of composite under boundary conditions in mechanical equilibrium 
        elif axis.lower() in ['vertical', 'vert', 'v']:
            # boundary conditions: uniaxial stress in vertical direction
            self.fe.bcbot(0.)  # fix bottom boundary
            self.fe.bcright(0., self.sides)  # boundary condition on lateral edges of model
            self.fe.bcleft(0., self.sides)
            self.fe.bctop(self.eps_tot * self.fe.leny, 'disp')  # strain applied to top nodes
            if self.sides == 'force':
                # fix lateral displacements of corner node to prevent rigid body motion
                hh = [no in self.fe.nobot for no in self.fe.noleft]
                noc = np.nonzero(hh)[0]  # find corner node
                self.fe.bcnode(noc, 0., 'disp', 'x')  # fix lateral displacement
            self.fe.solve()
        else:
            raise ValueError(f'Axis must be either "horizontal" or "vertical", not "{axis}"')
        # return stress and strain values
        return self.fe.sgl, self.fe.egl

    def calc_geom_param(self):
        """ Calculate volume fraction shape parameters of filler phase. The four shape parameters
        represent the minima and maxima of the x- and y-dimensions of voxel regions assigned to filler phase.
        """
        ind_x, ind_y = np.nonzero(self.el == 2)
        n_fil = len(ind_x)
        vf = n_fil / (self.NX*self.NY)  # volume fraction of filler phase
        ind_set = set()
        for i in range(n_fil):
            ind_set.add(f'{ind_x[i]}_{ind_y[i]}')
        lpx = []
        lpy = []
        for i in range(n_fil):
            nl = f'{ind_x[i]-1}_{ind_y[i]}' in ind_set
            nr = f'{ind_x[i]+1}_{ind_y[i]}' in ind_set
            nd = f'{ind_x[i]}_{ind_y[i]-1}' in ind_set
            nu = f'{ind_x[i]}_{ind_y[i]+1}' in ind_set
            if nl and nr and nd and nu:
                continue
            nc = 1
            if nl and not nr:
                while nl:
                    nc += 1
                    nl = f'{ind_x[i]-nc}_{ind_y[i]}' in ind_set
                lpx.append(nc)
                nc = 1
            if nu and not nd:
                while nu:
                    nc += 1
                    nu = f'{ind_x[i]}_{ind_y[i]+nc}' in ind_set
                lpy.append(nc)
                nc = 1
        shape = np.zeros((2,2))
        if len(lpx) > 0:
            shape[0, 0] = min(lpx)/self.NX
            shape[0, 1] = max(lpx)/self.NX
        if len(lpy) > 0:
            shape[1, 0] = min(lpy)/self.NY
            shape[1, 1] = max(lpy)/self.NY
        return vf, shape

    def upd_model(self, coords):
        p0 = coords[-2]
        p1 = coords[-1]
        ix = [int(p0[0] * self.NX / 10), int(p1[0] * self.NX / 10)]
        iy = [int(p0[1] * self.NY / 10), int(p1[1] * self.NY / 10)]
        NXi1 = min(ix)
        NXi2 = max(ix) + 1
        NYi1 = min(iy)
        NYi2 = max(iy) + 1
        self.el[NXi1:NXi2, NYi1:NYi2] = 2
        self.fe.mesh(elmts=self.el, NX=self.NX, NY=self.NY)  # create regular mesh with sections as defined in el
        return NXi1, NXi2, NYi1, NYi2

    def plot(self, val, mag=4, fig=None, ax=None):
        bar = False if val=='mat' else True
        fig, ax = self.fe.plot(val, mag=mag, shownodes=False, showbar=bar,
                               showfig=False, fig=fig, ax=ax)
        plt.show()
        self.p_set.add(fig)
        return fig, ax

    def close_plots(self):
        for p in self.p_set:
            plt.close(p)
        self.p_set = set()


class ResultDB(object):
    """ Database object for mechanical properties and geometric parameters of filler phase of different composites.
    Properties of matrix and filler phase must remain constant, only volume fraction and shapes may change!
    """
    def __init__(self, E1, E2):
        self.properties = {
            'Phase_Prop' : {
                'E1' : E1,
                'E2' : E2
            },
            'VF' : [],
            'SH_x0': [],
            'SH_x1': [],
            'SH_y0': [],
            'SH_y1': [],
            'Corners' : [],
            'E_vert' : [],
            'E_hor' : [],
        }
        
    def add(self, vf, shp, cpairs, Ev, Eh, E1, E2):
        if not (np.isclose(E1, self.properties['Phase_Prop']['E1']) and np.isclose(E1, self.properties['Phase_Prop']['E1'])):
            print('ERROR: wrong material definition.')
            print(f"This database is for composites with E1={self.properties['Phase_Prop']['E1']}"
                  f"and E2={self.properties['Phase_Prop']['E2']}, values provided are for E1={E1} and E2={E2}")
        self.properties['VF'].append(float(vf))
        self.properties['SH_x0'].append(float(shp[0, 0]))
        self.properties['SH_x1'].append(float(shp[0, 1]))
        self.properties['SH_y0'].append(float(shp[1, 0]))
        self.properties['SH_y1'].append(float(shp[1, 1]))
        self.properties['E_vert'].append(float(Ev))
        self.properties['E_hor'].append(float(Eh))
        hh = cpairs.ravel()
        self.properties['Corners'].append(len(hh))
        for val in hh:
            self.properties['Corners'].append(float(val))

    def plot(self):
        # calculate theoretical values
        E1 = self.properties['Phase_Prop']['E1']
        E2 = self.properties['Phase_Prop']['E2']
        Ev = np.array(self.properties['E_vert'])
        Eh = np.array(self.properties['E_hor'])
        vf = np.array(self.properties['VF'])
        shx = np.array(self.properties['SH_x1'])
        shy = np.array(self.properties['SH_y1'])

        # calculate theoretical homogenization values
        x = np.linspace(0.0, 1.0, 50)
        yV = (1.0 - x) * E1  + x * E2  # Voigt value for Young's modulus of composite
        yR = 1 / ((1.0 - x) / E1 + x / E2)  # Reuss value

        # set colormap for shapes
        n_val = len(vf)
        cm = plt.get_cmap('bwr')  # blue to white to red colormap

        # generate plot
        fig = plt.figure()
        plt.plot(x, yV, ':k')
        plt.plot(x, yR, '--k')
        for i, val in enumerate(vf):
            cind = shy[i] if shy[i] > 0.0 else 1
            plt.scatter(val, Ev[i], color=cm(0.5 + 0.5*cind))
            plt.scatter(val, Eh[i], color=cm(0.5 - 0.5*cind))
        #plt.colorbar()
        plt.xlabel('Filler volume fraction (.)')
        plt.ylabel("Eff. Young's modulus (GPa)")
        plt.legend(['Voigt rule', 'Reuss rule', 'E_vert', 'E_hor'])
        plt.title('Homogenized Stiffness')
        plt.show()
        return fig

    def write(self, fname=None, team=None):
        if fname is None:
            ttag = 'res_' if team is None else f'res_team{team}'
            fname = ttag + f"_{int(self.properties['Phase_Prop']['E1'])}_{int(elf.properties['Phase_Prop']['E2'])}.json"
        with open(fname, 'w') as fp:
            json.dump(self.properties, fp, indent=2)

    def read(self, fname, path='./'):
        with open(os.path.join(path, fname), 'r') as fp:
            res = json.load(fp)
        for key, val in self.properties.items():
            if key == 'Phase_Prop':
                if val['E1'] != res[key]['E1'] or val['E2'] != res[key]['E2']:
                    raise valueError('Phase properties in file do not match those of this database. Cannot import.')
                continue
            val += res[key]
            

def plot_comp(y_pred, y_ref, Emax, desc, ytest_pred=None, ytest_ref=None):
    x = np.linspace(0.0, Emax, 5, endpoint=True)
    fig = plt.figure()
    plt.plot(x, x, '-k', label='Reference')
    plt.plot(x, 1.1*x, ':k', label='+/- 10% error')
    plt.plot(x, 0.9*x, ':k')
    plt.plot(y_ref[:, 0], y_pred[:, 0], 'ro', label=desc + '_train, E_vert')
    plt.plot(y_ref[:, 1], y_pred[:, 1], 'bo', label=desc + '_train, E_horiz')
    if ytest_pred is not None and ytest_ref is not None:
        plt.plot(ytest_ref[:, 0], ytest_pred[:, 0],
                 linestyle='None', marker='o', color='#7A0000', label=desc + '_test, E_vert')
        plt.plot(ytest_ref[:, 1], ytest_pred[:, 1],
                 linestyle='None', marker='o', color='#00007A', label=desc + '_test, E_horiz')
    plt.xlabel("reference Young's modulus (GPa)")
    plt.ylabel("predicted Young's modulus (GPa)")
    plt.xlim(0.0, Emax+5)
    plt.ylim(0.0, Emax+5)
    plt.legend()
    plt.show()
    return fig


def extract_data(raw, n_feat=5, n_label=2):
    Ev = np.array(raw.properties['E_vert'])
    Eh = np.array(raw.properties['E_hor'])
    vf = np.array(raw.properties['VF'])
    shx0 = np.array(raw.properties['SH_x0'])
    shx1 = np.array(raw.properties['SH_x1'])
    shy0 = np.array(raw.properties['SH_y0'])
    shy1 = np.array(raw.properties['SH_y1'])
    n_data = len(vf)
    
    xd = np.zeros((n_data, n_feat))
    xd[:, 0] = vf
    xd[:, 1] = shx0
    xd[:, 2] = shx1
    xd[:, 3] = shy0
    xd[:, 4] = shy1

    yd = np.zeros((n_data, n_label))
    yd[:, 0] = Ev
    yd[:, 1] = Eh
    
    return xd, yd
