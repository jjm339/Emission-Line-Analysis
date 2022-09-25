"""
Julian's Notes:
    o For some reason scipy.optimize.curve_fit is unable to fit gaussians to the
      emission lines in the spectra, so we use functions and models from
      specutils and astropy.
          - specutils may or may not be already downloaded for some IDEs
    o The emission lines are currently being selected by manually scanning
      spectra and identifying noticeable emission lines.
    o The gaussian fitting currently only works on some emission lines when
      parameters are adjusted manually.

To do:
    1) Ensure the code works for most if not all observations.
    2) Figure out how to get the gaussian fitting to work for all emission lines
        - currently only works for specific lines when adjusted manually, this
          can be time consuming by having to check each fit plot to ensure it
          is working properly
    3) Obtain literature results of wavelength positions of emission lines
       and replace the peak_ref_wls setup with these values.
"""

"""
Import necessary packages
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D

from astropy import units as u
from astropy.modeling import models

fignum = 0  # For plotting purposes only
dir_fig = '/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/fig/20190809/'

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
Define functions
"""
def gauss(x,a,b,c):
    """
    Gauss function for use in gaussfit function.
    
    UNUSED FUNCTION
    """
    # a is amplitude, b is peak position, c is width
    # For our purposes we are only interested in b.

    return a*np.exp((-(x-b)**2)/(2*(c**2)))

def isnear(wavelength, reference_peaks, interval):
    """
    Checks if the wavelength input is near one of the reference peak
    wavelengths within a certain interval of wavelength.
    """
    for i in range(len(reference_peaks)):
        if abs(wavelength-reference_peaks[i]) < interval:
            return True
    return False

def gaussfit(wavelength, counts):
    """
    Performs gaussian fitting of emission line. Returns peak wavelength
    position and std deviation.
    IN:
        wavelength: 1D array of wavelength values
        counts: 1D array of counts values corresponding to wavelengths
    OUT:
        peak: peak position of emission line after gaussian fitting
        
    UNUSED FUNCTION
    """
    reduction = (np.mean(counts) + np.min(counts))/2
    popt, pcov = curve_fit(gauss,wavelength,counts-reduction, maxfev=50000)
    peak = popt[1]
    return peak

def get_mean(spectrum, wavelength):
    """
    Gets the wavelength position of the peak of the gaussian fit to the
    emission line.
    """
    peak = np.max(spectrum)
    index = 0
    for j in range(len(wavelength)):
        if y_fit[j] == peak:
            index = j
    return wavelength[index]

def fit_gauss(wl, cts, avg, ref_wl, precision):
    """
    Function version of gaussian fitting
    """
    spectrum = Spectrum1D(flux=(cts-avg)*u.ct, spectral_axis=wl*u.AA)
        
    fit_mean = ref_wl*u.AA
    fit_amp = (np.max(cts)-avg)*u.ct
        
    gauss_i = models.Gaussian1D(amplitude=fit_amp,mean=fit_mean,stddev=1*u.AA)
    gauss_fit = fit_lines(spectrum, gauss_i)
    x = np.linspace(wl[0],wl[-1],precision)
    y_fit = gauss_fit(x*u.AA)
    
    return y_fit, x

def index_range(wl, start, stop):
    a = 0
    b = 0
    for i in range(len(wl)):
        if wl[i] <= start:
            a = i
        if wl[i] <= stop:
            b = i
    return a, b

#%%
"""
Access data
"""

obs = 'b0076'

data1 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data1.npy'.format(obs))
data2 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data2.npy'.format(obs))

#%%
"""
Plot Spectra ORDER 1
"""
for i in range(len(data1)):
    plt.figure(fignum,figsize=(10,8))
    plt.plot(data1[i][0],data1[i][1])
    plt.title('Order 1, {} ({})'.format(obs,get_obj(obs,obj_dict)))
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    fignum += 1

#%%
"""
Plot Spectra (specific wavelength interval) ORDER 1
"""
for i in range(len(data1)):
    a, b = index_range(data1[i][0],8600,8650)
    plt.figure(fignum, figsize=(10,8))
    plt.plot(data1[i][0][a:b],data1[i][1][a:b])
    plt.title('Order 1, {} ({})'.format(obs,get_obj(obs,obj_dict)))
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    fignum += 1

#%%
"""
ORDER 1
Identify emission lines, extract arrays
"""
# Idea: identify reference emission lines that are clear among most of the
#       data sets and obtain peak wavelength positions across data sets through
#       gaussian fitting

test_peaks = find_peaks(data1[1][1],height=500)

peak_ref_wls = []
peak_ref_cts = []
for i in test_peaks[0]:
    peak_ref_wls.append(data1[1][0][i])
    peak_ref_cts.append(data1[1][1][i])

#----- Manual Choice -----#
#peak_ref_wls = [8399.208549499512, 8415.168960571289, 8430.22021484375, 8452.208290100098, 8465.280319213867, 8493.33169555664, 8504.61996459961, 8538.677276611328]
#peak_ref_cts = [6119.384638613443, 4102.442738517394, 10076.741082413439, 4971.529077872761, 6522.8591899904, 2920.5325500042118, 3291.576880044592, 965.8054206162901]
#-------------------------#
print(peak_ref_wls)
print(peak_ref_cts)

print(test_peaks[0])

savefig = False

plt.figure(fignum,figsize=(10,8))
plt.plot(data1[1][0],data1[1][1])
plt.scatter(peak_ref_wls,peak_ref_cts,color='r',label='Chosen Emission Lines')
plt.title('{} (Order 1) Visualization of Chosen Emission Lines'.format(obs))
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Counts')
plt.legend(loc='upper right')
if savefig:
    plt.savefig('{}/{}_Chosen_Lines_1.png'.format(dir_fig,obs), format='png')
fignum += 1

# Ideally the reference wavelengths of emission lines would be input manually
# taken from results of literature. Right now the code above must be adjusted
# manually for each different observation.

#print(peak_ref_wls)
interval = 1.25

data1_wls = []
data1_cts = []
for j in range(len(data1)):
    wls = []
    cts = []
    temp_wl = []
    temp_cts = []
    for k in range(len(data1[j][0])):
       # This loop isolates intervals of wavelength around each reference
       # wavelength point.
       if isnear(data1[j][0][k], peak_ref_wls, interval):
           temp_wl.append(data1[j][0][k])
           temp_cts.append(data1[j][1][k])
       else:
           if len(temp_wl) != 0:
               wls.append(temp_wl)
               cts.append(temp_cts)
               temp_wl, temp_cts = [], []
    data1_wls.append(wls)
    data1_cts.append(cts)

#%%
"""
Data refinement (deal with any NAN values in data1_cts)
"""
for i in range(len(data1_cts)):
    for j in range(len(data1_cts[i])):
        if np.isnan(data1_cts[i][j]).any():
            for a in range(len(data1_cts[i][j])):
                if np.isnan(data1_cts[i][j][a]):
                    data1_cts[i][j][a] = \
                        sum((data1_cts[i][j][a-1],data1_cts[i][j][a+1]))/2

#%%
"""
ORDER 1
Gauss fitting to extracted wavelengths
"""
line_num = 7

dist = interval/4

plot = False
precision = int(round(100*interval))
lines = []
for i in range(len(data1)):
    temp_lines = []
    #for j in range(len(peak_ref_wls)):
    for j in range(line_num, line_num+1):
        # This loop fits the previously extracted intervals of data and fits
        # a gaussian curve to find the wavelength position of the emission line.
        
        avg = (data1_cts[i][j][0] + data1_cts[i][j][-1])/2   
        y_fit, x = fit_gauss(data1_wls[i][j],data1_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = sum((np.mean(data1_cts[i][j]),np.min(data1_cts[i][j])))/2
            y_fit, x = fit_gauss(data1_wls[i][j],data1_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed again, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = np.mean(data1_cts[i][j])
            y_fit, x = fit_gauss(data1_wls[i][j],data1_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed again, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = np.min(data1_cts[i][j])
            y_fit, x = fit_gauss(data1_wls[i][j],data1_cts[i][j],avg,peak_ref_wls[j],precision)
        
        if plot:
            plt.figure(fignum,figsize=(10,8))
            
            plt.plot(data1_wls[i][j],data1_cts[i][j],label='Spectrum')
            plt.plot(x,y_fit+(avg*u.ct),label='Fit')
            plt.legend(loc='upper right')
            plt.title('Emission Line fit of {} ({}), Spectrum #{} (Order 1)'.format(obs,get_obj(obs,obj_dict),i+1))
            
            fignum += 1
        
        temp_lines.append(get_mean(y_fit,x))
    lines.append(temp_lines)

#%%
"""
ORDER 1
Plot results with histograms, fit gaussian
"""
savefig = False

lines = np.array(lines)
for i in range(len(lines[0])):
    plt.figure(fignum, figsize=(10,8))
    n, bins, pat = plt.hist(lines[:,i], bins=round(np.sqrt(len(lines[:,i]))),\
                            label='Fitted Peaks')
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    plt.title('{} (Order 1) Histogram of Emission Line (near {} $\AA$) Wavelength Position with {} bins'.format(obs,round(peak_ref_wls[line_num]),len(n)))

    x = []
    for i in range(len(bins)-1):
        x.append((bins[i]+bins[i+1])/2)

    spectrum = Spectrum1D(flux=n*u.ct, spectral_axis=x*u.AA)
    fit_mean = ((bins[0]+bins[-1])/2)*u.AA
    fit_amp = np.max(n)*u.ct
    
    #----- Manual Input -----#
    #fit_mean = 8504.85*u.AA
    #fit_amp = 1*u.ct
    #------------------------#
    
    gauss_i = models.Gaussian1D(amplitude=fit_amp,mean=fit_mean,stddev=0.01*u.AA)
    gauss_fit = fit_lines(spectrum, gauss_i)
    x2 = np.linspace(bins[0],bins[-1],1000)
    y_fit = gauss_fit(x2*u.AA)

    plt.plot(x2,y_fit,label='Gauss Fit')
    plt.legend()
    plt.text(x2[0],np.max(y_fit).value-1,'Mean: {} $\AA$ \n$\sigma$: {} $\AA$'.format(gauss_fit.mean.value,gauss_fit.stddev.value))
    
    if savefig:
        plt.savefig('{}/{}_Em_Line_{}_Hist.png'.format(dir_fig,obs,round(peak_ref_wls[line_num])), format='png')
    
    fignum += 1
    
    print(gauss_fit)

#%%
"""
Plot Spectra ORDER 2
"""
for i in range(len(data2)):
    plt.figure(fignum,figsize=(10,8))
    plt.plot(data2[i][0],data2[i][1])
    plt.title('Order 2, {} ({})'.format(obs,get_obj(obs,obj_dict)))
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    fignum += 1

#%%
"""
Plot Spectra (specific wavelength interval) ORDER 2
"""
for i in range(len(data2)):
    a, b = index_range(data2[i][0],8620,8630)
    plt.figure(fignum, figsize=(10,8))
    plt.plot(data2[i][0][a:b],data2[i][1][a:b])
    plt.title('Order 2, {} ({})'.format(obs,get_obj(obs,obj_dict)))
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    fignum += 1

#%%
"""
ORDER 2
Identify emission lines, extract arrays
"""
# Idea: identify reference emission lines that are clear among most of the
#       data sets and obtain peak wavelength positions across data sets through
#       gaussian fitting

test_peaks = find_peaks(data2[0][1], height=500)
#print(test_peaks)

peak_ref_wls = []
peak_ref_cts = []
for i in test_peaks[0]:
    peak_ref_wls.append(data2[0][0][i])
    peak_ref_cts.append(data2[0][1][i])

#----- Manual Choice -----#
peak_ref_wls = [8624.533073425293,8631.488906860352,8660.63372039795,8665.553329467773,8758.699890136719,8761.337219238281,8767.925170898438, 8776.239990234375, 8778.322143554688]
peak_ref_cts = [939.7917228717317,1139.152137937255,1456.3894640199587,1336.2959435545288,1781.559083669806,3596.4757724595183,5280.2190396966225, 2098.4651166215435, 4755.115775962172]
#-------------------------#
print(peak_ref_wls)
print(peak_ref_cts)

savefig = True

plt.figure(fignum,figsize=(10,8))
plt.plot(data2[0][0],data2[0][1])
plt.scatter(peak_ref_wls,peak_ref_cts,color='r',label='Chosen Emission Lines')
plt.title('{} (Order 2) Visualization of Chosen Emission Lines'.format(obs))
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Counts')
plt.legend(loc='upper left')
if savefig:
    plt.savefig('{}/{}_Chosen_Lines_2.png'.format(dir_fig,obs), format='png')
fignum += 1

# Ideally the reference wavelengths of emission lines would be input manually
# taken from results of literature. Right now the code above must be adjusted
# manually for each different observation.

#print(peak_ref_wls)
interval = 0.9

data2_wls = []
data2_cts = []
for j in range(len(data2)):
    wls = []
    cts = []
    temp_wl = []
    temp_cts = []
    for k in range(len(data2[j][0])):
       # This loop isolates intervals of wavelength around each reference
       # wavelength point.
       if isnear(data2[j][0][k], peak_ref_wls, interval):
           temp_wl.append(data2[j][0][k])
           temp_cts.append(data2[j][1][k])
       else:
           if len(temp_wl) != 0:
               wls.append(temp_wl)
               cts.append(temp_cts)
               temp_wl, temp_cts = [], []
    data2_wls.append(wls)
    data2_cts.append(cts)

#%%
"""
Data refinement (deal with any NAN values in data2_cts)
"""
for i in range(len(data2_cts)):
    for j in range(len(data2_cts[i])):
        if np.isnan(data2_cts[i][j]).any():
            for a in range(len(data2_cts[i][j])):
                if np.isnan(data2_cts[i][j][a]):
                    data2_cts[i][j][a] = \
                        sum((data2_cts[i][j][a-1],data2_cts[i][j][a+1]))/2
    
#%%
"""
ORDER 2
Gauss fitting to extracted wavelengths
"""
line_num = 8

dist = interval/4

plot = True
precision = int(round(100*interval))
lines = []
for i in range(len(data2)):
    temp_lines = []
    #for j in range(len(peak_ref_wls)):
    for j in range(line_num, line_num+1):
        # This loop fits the previously extracted intervals of data and fits
        # a gaussian curve to find the wavelength position of the emission line.
        
        avg = (data2_cts[i][j][0] + data2_cts[i][j][-1])/2   
        y_fit, x = fit_gauss(data2_wls[i][j],data2_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = (np.mean(data2_cts[i][j]) + np.min(data2_cts[i][j]))/2
            y_fit, x = fit_gauss(data2_wls[i][j],data2_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed again, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = np.mean(data2_cts[i][j])
            y_fit, x = fit_gauss(data2_wls[i][j],data2_cts[i][j],avg,peak_ref_wls[j],precision)
        
        # Check if fit failed again, try different avg parameter
        if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
            avg = np.min(data2_cts[i][j])
            y_fit, x = fit_gauss(data2_wls[i][j],data2_cts[i][j],avg,peak_ref_wls[j],precision)
        
        if plot:
            plt.figure(fignum,figsize=(10,8))
            
            plt.plot(data2_wls[i][j],data2_cts[i][j],label='Spectrum')
            plt.plot(x,y_fit+(avg*u.ct),label='Fit')
            plt.legend(loc='upper right')
            plt.title('Emission Line fit of {} ({}), Spectrum #{} (Order 2)'.format(obs,get_obj(obs,obj_dict),i+1))
            
            fignum += 1
        
        temp_lines.append(get_mean(y_fit,x))
    lines.append(temp_lines)

#%%
"""
ORDER 2
Plot results with histograms, fit gaussian
"""
savefig = True

lines = np.array(lines)
for i in range(len(lines[0])):
    plt.figure(fignum, figsize=(10,8))
    n, bins, pat = plt.hist(lines[:,i], bins=round(np.sqrt(len(lines[:,i]))),\
                            label='Fitted Peaks')
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    plt.title('{} (Order 2) Histogram of Emission Line (near {} $\AA$) Wavelength Position with {} bins'.format(obs,round(peak_ref_wls[line_num]),len(n)))

    x = []
    for i in range(len(bins)-1):
        x.append((bins[i]+bins[i+1])/2)

    spectrum = Spectrum1D(flux=n*u.ct, spectral_axis=x*u.AA)
    fit_mean = ((bins[0]+bins[-1])/2)*u.AA
    fit_amp = np.max(n)*u.ct
    
    #----- Manual Input -----#
    #fit_mean = 8758.725*u.AA
    #fit_amp = 1*u.ct
    #------------------------#
    
    gauss_i = models.Gaussian1D(amplitude=fit_amp,mean=fit_mean,stddev=0.01*u.AA)
    gauss_fit = fit_lines(spectrum, gauss_i)
    x2 = np.linspace(bins[0],bins[-1],1000)
    y_fit = gauss_fit(x2*u.AA)

    plt.plot(x2,y_fit,label='Gauss Fit')
    plt.legend()
    plt.text(x2[0],np.max(y_fit).value-1,'Mean: {} $\AA$ \n$\sigma$: {} $\AA$'.format(gauss_fit.mean.value,gauss_fit.stddev.value))
    
    if savefig:
        plt.savefig('{}/{}_Em_Line_{}_Hist.png'.format(dir_fig,obs,round(peak_ref_wls[line_num])), format='png')
    
    
    fignum += 1
    
    print(gauss_fit)