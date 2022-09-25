"""
Julian's Notes:
    The contourf plots are only to be used as an initial view of the spectra
    and their defining features. The imshow plots are more detailed and are
    better for qualitative analysis of emission lines than the contourf plots.
    
To do:
    1) Create more visualization methods of emission lines.
        - Panel views for each or multiple emission lines
        - Squished view of individual emission lines
"""

"""
Import necessary packages
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

fignum = 0

#%%
"""
Define functions
"""
def get_x(x, spectrum):
    """
    Finds indeces of x array for interpolation for spectrum data array. For use
    in interp_spectra function.
    IN:
        x: common 1d wavelength grid
        spectrum: wavelength array of spectrum
    OUT:
        start and stop indeces for x array for interpolation
    """
    a, b = None, None
    for i in range(len(x)):
        if x[i] >= spectrum[0] and (a is None):
            a = i
        if x[i] >= spectrum[-1] and (b is None):
            b = i-1
    
    return a,b   

def interp_spectra(data, length, method):
    """
    Interpolates the spectra data to a common wavelength grid. This is
    necessary for spectra with different wavelength ranges and/or coordinate
    points if you want to plot the data with plt.imshow().
    IN:
        data: the array of spectra wavelength and value arrays
        length (int): the length of the 1d grid (wavelength axis)
        method (str): interpolation kind, 'linear' or 'cubic'
    OUT:
        new_data: a new 2d array of interpolated data
        extent: beginning and ending wavelength
    """
    # Check for incorrect method input
    if not((method == 'linear') or (method == 'cubic')):
        raise ValueError("Method entered is not \'linear\' or \'cubic\'.")
    
    # Create common 1D wavelength grid
    min_wl, max_wl = data[0][0][0], data[0][0][-1]
    for i in range(1,len(data)):
        if data[i][0][0] < min_wl:
            min_wl = data[i][0][0]
        if data[i][0][-1] > max_wl:
            max_wl = data[i][0][-1]
    x = np.linspace(min_wl, max_wl, length)
    
    extent = [x[0], x[-1],len(data),0]  # For axis labels of plt.imshow()
    
    new_data = []
    for i in range(len(data)):
        # Initialize interpolation for specific spectrum.
        interp = interpolate.interp1d(data[i][0], data[i][1], kind=method)
        
        # Spectra have different ranges of wavelenghts within x, so find
        # range of indeces to interpolate data in (get_x function).
        a,b = get_x(x,data[i][0])
        
        # Interpolate data. Entries outside of spectrum range are set to 0.
        new_y = np.zeros(len(x))
        temp_y = interp(x[a:b])
        for j in range(a,b):
            new_y[j] = temp_y[j-a]
        new_data.append([x,new_y])
    return new_data, extent

#%%
"""
Access Observations to Objects Dictionary
"""
exec(open('obj_dict.py').read())    # Imports dictionary where the keys are
                                    # the observation numbers in the excel
                                    # sheet and the items are the objects.
                                    # Also imports get_obj() function.
#print(obj_dict)    # Verify dictionary is imported properly

#%%
"""
Access data
"""
# Can only access one observation at a time
obs = 'b0050'

data1 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data1.npy'.format(obs))
data2 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data2.npy'.format(obs))

#%%
"""
Plot left side with contourf
"""
# Create 2D X, Y, and Z grids.
yl = []
for i in range(len(data1)):
    temp_yl = []
    for j in range(len(data1[i][0])):
        temp_yl.append(i)
    yl.append(temp_yl)

xl = []
zl = []
for i in range(len(data1)):
    temp_xl = []
    temp_zl = []
    for j in range(len(data1[i][0])):
        temp_xl.append(data1[i][0][j])
        temp_zl.append(data1[i][1][j])
    xl.append(temp_xl)
    zl.append(temp_zl)

# Plot
plt.figure(fignum,(6,5))
fignum += 1
plt.contourf(xl,yl,zl,levels=100, cmap='gray')
cbar = plt.colorbar()
cbar.set_label('Counts')
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Spectrum')
plt.title('Visualization of Spectra (Order 1) of {} ({})'.format(obs,get_obj(obs,obj_dict)))

#%%
"""
Plot left side with imshow
"""
# First interpolate data to a common wavelength grid
method = 'linear'   # 'linear' or 'cubic'. Other options will raise an error.
data_l, extent_l = interp_spectra(data1, 1928, method)

# Obtain z 2d array and plot
zl2 = []
for i in range(len(data_l)):
    temp_zl2 = []
    for j in range(len(data_l[i][0])):
        temp_zl2.append(data_l[i][1][j])
    zl2.append(temp_zl2)

vmin_l = 0      # 0 is fine
vmax_l = 4000   # 4000 is typical value, 2000 is brightened

plt.figure(fignum,(20,8))
fignum += 1
plt.imshow(zl2,cmap='gray', extent=extent_l, vmin=vmin_l, vmax=vmax_l)
plt.title('Visualization of Spectra (Order 1) of {} ({}) Using {} Interpolation'.format(obs,get_obj(obs,obj_dict),method))
plt.ylabel('Spectrum')
plt.xlabel('Wavelength ($\AA$)')

#%%
"""
Plot right side with contourf
"""
# Create 2D X, Y, and Z grids.
yr = []
for i in range(len(data2)):
    temp_yr = []
    for j in range(len(data2[i][0])):
        temp_yr.append(i)
    yr.append(temp_yr)

xr = []
zr = []
for i in range(len(data2)):
    temp_xr = []
    temp_zr = []
    for j in range(len(data2[i][0])):
        temp_xr.append(data2[i][0][j])
        temp_zr.append(data2[i][1][j])
    xr.append(temp_xr)
    zr.append(temp_zr)

# Plot
plt.figure(fignum,(6,5))
fignum += 1
plt.contourf(xr,yr,zr,levels=100, cmap='gray')
cbar = plt.colorbar()
cbar.set_label('Counts')
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Spectrum')
plt.title('Visualization of Spectra (Order 2) of {} ({})'.format(obs,get_obj(obs,obj_dict)))

#%%
"""
Plot right side with imshow
"""
# First interpolate data to a common wavelength grid
method = 'linear'   # 'linear' or 'cubic'. Other options will raise an error.
data_r, extent_r = interp_spectra(data2, 1928, method)

# Obtain z 2d array and plot
zr2 = []
for i in range(len(data_r)):
    temp_zr2 = []
    for j in range(len(data_r[i][0])):
        temp_zr2.append(data_r[i][1][j])
    zr2.append(temp_zr2)

vmin_r = 0      # 0 is fine
vmax_r = 500   # 4000 is typical value, 2000 is brightened

plt.figure(fignum,(20,8))
fignum += 1
plt.imshow(zr2,cmap='gray', extent=extent_r, vmin=vmin_r, vmax=vmax_r)
plt.title('Visualization of Spectra (Order 2) of {} ({}) Using {} Interpolation'.format(obs,get_obj(obs,obj_dict),method))
plt.ylabel('Spectrum')
plt.xlabel('Wavelength ($\AA$)')