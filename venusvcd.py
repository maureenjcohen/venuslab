# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## File paths ##
# %%
prof1 = '/home/maureenjcohen/misc_data/VCD/VCD_0lon_0lat_12lt.txt'
prof2 = '/home/maureenjcohen/misc_data/VCD/VCD_180lon_0lat_12lt.txt'

# %%
class VCD:
    """ Data object to read in and store Venus Climate Database
    profiles for atmospheric temperature, zonal wind, and scale hight """

    def __init__(self, profpath, name):
        self.name = name

        temp = pd.read_csv(profpath, names=['Altitude','Temperature'], header=None, skiprows=9, nrows=35, sep='\s+')
        u = pd.read_csv(profpath, names=['Altitude','Zonal wind'], header=None, skiprows=53, nrows=35, sep='\s+')
        H = pd.read_csv(profpath, names=['Altitude','Scale height'], header=None, skiprows=97, nrows=35, sep='\s+')

        labels = {'Altitude':'m',
                  'Zonal wind':'m/s',
                  'Temperature':'K',
                  'Scale height':'m'}
        self.labels = labels # Store labels dictionary in object
        cols = labels.keys() # Extract column labels to be used in header

        data = pd.DataFrame(columns=cols)
        data['Altitude'] = temp['Altitude'].values
        data['Temperature'] = temp['Temperature'].values
        data['Zonal wind'] = u['Zonal wind'].values
        data['Scale height'] = H['Scale height'].values

        self.data = data # Add DataFrame to object
        units = labels.values() # Extract units from labels
        self.units = units # Add units to object
        self.heights = data['Altitude'].values*1e-3 # Add heights in km

    def profile(self, key):
        if key=='Zonal wind':
            cube = self.data['Zonal wind'].values
        elif key=='Temperature':
            cube = self.data['Temperature'].values
        elif key=='Scale height':
            cube = self.data['Scale height'].values
        else:
            print('Key not recognised. Choose Pressure or Temperature')

        fig, ax = plt.subplots(figsize=(8,6))
        plt.plot(cube, self.data['Altitude']*1e-3)
        plt.title(f'{key} profile from {self.name}')
        plt.xlabel(f'{key} [{self.labels[key]}]')
        plt.ylabel('Altitude [km]')
        plt.show()


# %%
