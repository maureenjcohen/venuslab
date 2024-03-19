""" Module for processing Vega data """

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## File paths ##
# %%
balloon1 = '/exomars/data/analysis/volume_8/mc5526/vega_data/vg1bl_rdr.dat'
balloon2 = '/exomars/data/analysis/volume_8/mc5526/vega_data/vg2bl_rdr.dat'
lander2 = '/exomars/data/analysis/volume_8/mc5526/vega_data/vg2lr.dat'

# %%
class Balloon:
    """ Holds data from a Vega balloon experiment
        Pandas DataFrame                        """

    def __init__(self, balloonpath, name):
        self.name = name # Name of experiment, i.e. 'Vega 1' or 'Vega 2'

        data = pd.read_csv(balloonpath, header=None, sep='\s+')

        labels = {'Time':'s', 
                  'Pressure':'hPa', 'Pressure flag':'int',
                  'Temperature':'K', 'Temperature flag': 'int',
                  'Entropy':'J/K', 'Entropy flag':'int',
                  'Backscatter': '10^-8 /M/SR', 'Backscatter flag':'int',
                  'Effective radius':'km', 'Radius flag':'int',
                  'Latitude':'rad', 'Latitude flag':'int',
                  'Longitude':'rad', 'Longitude flag':'int',
                  'U':'m/s', 'U flag':'int',
                  'V':'m/s', 'V flag':'int',
                  'W_a':'m/s', 'W_a flag':'int',
                  'W_b':'m/s', 'W_b flag':'int'}
        # Labels data taken from vg1bl_rdr.lbl and vg2bl_rdr.lbl

        self.labels = labels # Store labels dictionary in object
        cols = labels.keys() # Extract column labels to be used in header
        data.columns = cols # Add column names to DataFrame
        self.data = data # Add DataFrame to object

        units = labels.values() # Extract units from labels
        self.units = units # Add units to object

    def explain_flags(self):
        print('Data quality flag for previous column.')
        print('0 indicates no data; 1 indicates erroneous data.')
        print('2 or 3 indicate good data, but some uncertainties.') 
        print('4 indicates high quality.')

    def coords(self):
        """ Define coords of ballon as mean pressure level
            and mean latitude; longitude varies with drift """
        
        pmean = np.mean(self.data['Pressure'])
        latclean = self.data['Latitude']
        latclean[latclean==0] = np.nan
        latmean = np.rad2deg(np.mean(latclean))

        self.lat = latmean # Mean latitude of balloon
        self.plev = pmean # Mean pressure level of balloon

    def calc_stats(self):
        """ Get mean U, V, W winds from whole dataset """

        wmean = np.mean(self.data['W_b'])
        uclean = self.data['U']
        uclean[uclean==0] = np.nan
        umean = np.mean(uclean)
        vclean = self.data['V']
        vclean[vclean==0] = np.nan
        vmean = np.mean(vclean)

        stats_data = {'Field': ['U', 'V', 'W'],
                      'Mean': [umean, vmean, wmean]}
        stats_df = pd.DataFrame(stats_data)
        self.stats = stats_df
        # Create DF with mean U, V, W winds

# %%
class Descent:
    """ Holds data from descent probes 
        Currently handles: Vega 2 Lander    """
    
    def __init__(self, landerpath, name):
        self.name = name # Name of experiment

        data = pd.read_csv(landerpath, header=None, sep='\s+')
        del data[data.columns[0]]

        labels = {'Time': 's',
                  'Pressure': 'bar',
                  'Temperature': 'K',
                  'Altitude': 'm'}
        # Labels data taken from vg2lr.lbl

        self.labels = labels # Store labels dictionary in object
        cols = labels.keys() # Extract column labels to be used in header
        data.columns = cols # Add column names to DataFrame
        self.data = data # Add DataFrame to object

        units = labels.values() # Extract units from labels
        self.units = units # Add units to object

    def profile(self, key):
        if key=='Pressure':
            cube = self.data['Pressure'].values
        elif key=='Temperature':
            cube = self.data['Temperature'].values
        else:
            print('Key not recognised. Choose Pressure or Temperature')

        fig, ax = plt.subplots(figsize=(8,6))
        plt.plot(cube, self.data['Altitude']*1e-3)
        plt.title(f'{key} profile from {self.name}')
        plt.xlabel(f'{key} [{self.labels[key]}]')
        plt.ylabel('Altitude [km]')
        plt.show()   


# %%
def compare_stats(plobject, bobject, lat, lev=25):

    """ Compare mean wind field values of a 
        simulation output and a Vega balloon 
        
        Vega 1: lat = 8 deg (~model 52), lev = 580 hPa (~model 25)
        Vega 2: lat = """
    
    if not hasattr(bobject, 'stats'):
        bobject.calc_stats()
    
    sim_u = np.mean(plobject.data['vitu'][:,lev,lat,:])
    sim_v = np.mean(plobject.data['vitv'][:,lev,lat,:])

    omega = plobject.data['vitw'][:,lev,lat,:]
    temp = plobject.data['temp'][:,lev,lat,:]
    pres = plobject.data['presnivs'][lev]
    w = -(omega*temp*plobject.RCO2)/(pres*plobject.g)
    sim_w = np.mean(w)

    compstats = {'Field': ['U', 'V', 'W'],
                'Mean': list(bobject.stats['Mean'].values),
                'Sim mean': [sim_u, sim_v, sim_w]}
    compdf = pd.DataFrame(compstats)
    
    print('Comparison of wind fields at approx.:') 
    print(f'lat {plobject.lats[lat]} deg,')
    print(f'altitude {plobject.heights[lev]} km')

    print(compdf.head)
    
    return compdf   


# %%
