"""
Julian's Notes:
    o For some reason the plotting works properly and no plots are overlapped,
      other times multiple plots jumble up into horrible messes. No idea why.
    o The directories in the code will have to be changed for each user.
"""

"""
Import necessary packages
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D

from astropy import units as u
from astropy.modeling import models

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

"""
Define functions
"""
def gauss(x,a,b,c):
    """
    Gaussian model function
    """
    # a is amplitude, b is peak position, c is width
    return a*np.exp((-(x-b)**2)/(2*(c**2)))

def isnear(wavelength, reference_peak, interval):
    """
    Checks if the wavelength input is near one of the reference peak
    wavelengths within a certain interval of wavelength.
    """
    if abs(wavelength-reference_peak) < interval:
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
        if spectrum[j] == peak:
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

def stddev(data, mean):
    """
    Calculate stddev of data
    """
    n = len(data)
    numerator = 0
    for i in range(n):
        numerator += (data[i]-mean)**2
    return np.sqrt(numerator/n)

#%%
"""
Define Main Functions
"""
def get_lines(wls, cts, order, plot, fignum):
    """
    Return arrays of reference emission line wavelengths and cts for use for
    all spectra
    """
    
    ref_lines_1 = [8399, 8415, 8430, 8452, 8465, 8493, 8504, 8538]  # Order 1
    ref_lines_2 = [8624, 8631, 8660, 8665, 8758, 8761, 8767, 8776, 8778]    # Order 2
    
    test_pks = find_peaks(cts,height=500)
    
    ref_pks = []
    ref_wls = []
    for i in test_pks[0]:
        if order == 1:
            if int(wls[i]) in ref_lines_1:
                ref_wls.append(wls[i])
                ref_pks.append(cts[i])
        elif order == 2:
            if int(wls[i]) in ref_lines_2:
                ref_wls.append(wls[i])
                ref_pks.append(cts[i])
    if plot:
        plt.figure(fignum,figsize=(10,8))
        plt.plot(wls,cts)
        plt.scatter(ref_wls,ref_pks,color='r',label='Chosen Emission Lines')
        plt.title('{} (Order 1) Visualization of Chosen Emission Lines'.format(obs))
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Counts')
        plt.legend(loc='upper right')
        plt.savefig('{}/{}_Chosen_Lines_{}.png'.format(dir_fig,obs,order), format='png')
        fignum += 1
    return ref_wls, ref_pks, fignum
    
def get_line_arrays(data, peaks, intervals):
    """
    Return arrays of wavelengths and cts around each emission line for all
    observation spectra
    """
    data_wls = []
    data_cts = []
    for j in range(len(data)):
        # This loop extracts intervals of data around each emission line in
        # each spectrum.
        wls = []
        cts = []
        for i in range(len(peaks)):
            temp_wl = []
            temp_cts = []
            for k in range(len(data[j][0])):
                if isnear(data[j][0][k], peaks[i], intervals[i]):
                    temp_wl.append(data[j][0][k])
                    temp_cts.append(data[j][1][k])
                else:
                    if len(temp_wl) != 0:
                        wls.append(temp_wl)
                        cts.append(temp_cts)
                        temp_wl, temp_cts = [], []
        data_wls.append(wls)
        data_cts.append(cts)
    return data_wls, data_cts

def remove_nans(data_wls, data_cts):
    """
    Remove NAN points from data
    """
    # Find indexes where value is NaN and remove those points
    new_wls, new_cts = [], []
    for i in range(len(data_cts)):
        temp_wl, temp_ct = [], []
        for j in range(len(data_cts[i])):
            tmp_wl, tmp_ct = [], []
            for k in range(len(data_cts[i][j])):
                if not np.isnan(data_cts[i][j][k]):
                    tmp_wl.append(data_wls[i][j][k])
                    tmp_ct.append(data_cts[i][j][k])
            temp_wl.append(tmp_wl)
            temp_ct.append(tmp_ct)
        new_wls.append(temp_wl)
        new_cts.append(temp_ct)
    return new_wls, new_cts

def fitting(data, data_wls, data_cts, peak_ref_wls, intervals, plot, fignum):
    """
    Perform the gaussian fitting of each emission line in each spectrum
    """
    fignum = 1
    lines = []
    for i in range(len(data)):
        temp_lines = []
        for j in range(len(peak_ref_wls)):
            # This loop fits the previously extracted intervals of data and fits
            # a gaussian curve to find the wavelength position of the emission line.
            
            dist = intervals[j]/4
            precision = int(round(100*intervals[j]))
        
            avg = (data_cts[i][j][0] + data_cts[i][j][-1])/2   
            y_fit, x = fit_gauss(data_wls[i][j],data_cts[i][j],avg,peak_ref_wls[j],precision)
        
            # Check if fit failed, try different avg parameter
            if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
                avg = sum((np.mean(data_cts[i][j]),np.min(data_cts[i][j])))/2
                y_fit, x = fit_gauss(data_wls[i][j],data_cts[i][j],avg,peak_ref_wls[j],precision)
        
            # Check if fit failed again, try different avg parameter
            if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
                avg = np.mean(data_cts[i][j])
                y_fit, x = fit_gauss(data_wls[i][j],data_cts[i][j],avg,peak_ref_wls[j],precision)
        
            # Check if fit failed again, try different avg parameter
            if abs(get_mean(y_fit,x)-peak_ref_wls[j]) >= dist:
                avg = np.min(data_cts[i][j])
                y_fit, x = fit_gauss(data_wls[i][j],data_cts[i][j],avg,peak_ref_wls[j],precision)
        
            if plot:
                plt.figure(fignum,figsize=(10,8))
            
                plt.plot(data_wls[i][j],data_cts[i][j],label='Spectrum')
                plt.plot(x,y_fit+(avg*u.ct),label='Fit')
                plt.legend(loc='upper right')
                plt.title('Emission Line fit of {} ({}), Spectrum #{} (Order 1)'.format(obs,get_obj(obs,obj_dict),i+1))
                plt.savefig('{}/{}_Line_Fit_{}_{}.png'.format(dir_fig,obs,i,j), format='png')
                fignum += 1
        
            temp_lines.append(get_mean(y_fit,x))
        lines.append(temp_lines)
    lines = np.array(lines)
    return lines, fignum

def data_clip(lines,low,high):
    """
    Perform sigma clipping on data sets.
    Also reorganizes the way the data is stored.
    """
    clipped = []
    for i in range(len(lines[0])):
        temp_clipped, a, b = sigmaclip(lines[:,i], low, high)
        clipped.append(temp_clipped)
    return clipped

def histograms_old(lines, peak_ref_wls, order, binnum, fignum):
    """
    OLD VERSION (Does gaussian fitting of histogram, has no binsize 
                 customization parameter)
    Perform histograms of results & fits
    """
    results = []
    for i in range(len(peak_ref_wls)):
        result = []
        plt.figure(fignum, figsize=(10,8))
        n, bins, pat = plt.hist(lines[i], bins=binnum, label='Fitted Peaks')
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Counts')
        plt.title('{} (Order {}) Histogram of Emission Line (near {} $\AA$) Wavelength Position with {} bins'.format(obs,order,round(peak_ref_wls[i]),len(n)))

        x = []
        for j in range(len(bins)-1):
            x.append((bins[j]+bins[j+1])/2)

        spectrum = Spectrum1D(flux=n*u.ct, spectral_axis=x*u.AA)
        fit_mean = ((bins[0]+bins[-1])/2)*u.AA
        fit_amp = np.max(n)*u.ct
    
        gauss_i = models.Gaussian1D(amplitude=fit_amp,mean=fit_mean,stddev=0.01*u.AA)
        gauss_fit = fit_lines(spectrum, gauss_i)
        x2 = np.linspace(bins[0],bins[-1],1000)
        y_fit = gauss_fit(x2*u.AA)

        plt.plot(x2,y_fit,label='Gauss Fit')
        plt.legend()
        plt.text(x2[0],np.max(y_fit).value-1,'Mean: {} $\AA$ \n$\sigma$: {} $\AA$'.format(gauss_fit.mean.value,gauss_fit.stddev.value))

        plt.savefig('{}/{}_Em_Line_{}_Hist.png'.format(dir_fig,obs,round(peak_ref_wls[i])), format='png')
    
        fignum += 1
        
        result = [gauss_fit.mean.value,gauss_fit.stddev.value]
        results.append(result)
    results = np.array(results)
    return results, fignum

def histograms(lines, peak_ref_wls, order, binnum, plot, fignum):
    """
    Perform histograms of results & fits
    """  
    results = []
    for i in range(len(peak_ref_wls)):
        # Plot histogram
        plt.figure(fignum, figsize=(10,8))
        n, bins, pat = plt.hist(lines[i], bins=binnum, label='Fitted Peaks')
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Counts')
        plt.title('{} (Order {}) Histogram of Emission Line (near {} $\AA$) Wavelength Position with {} bins'.format(obs,order,round(peak_ref_wls[i]),len(n)))
        
        # Calculate mean & stddev, store this data
        data_mean = np.mean(lines[i])
        data_stddev = stddev(lines[i], data_mean)
        results.append([data_mean, data_stddev])
        
        x = np.linspace(bins[0], bins[-1], 100)
        
        # Plot gaussian on histogram, using calculated mean & stddev
        plt.plot(x,gauss(x,np.max(n),data_mean,data_stddev),\
                 label = 'Gauss Curve', color='orange')
        plt.legend()
        if plot:
            plt.savefig('{}/{}_Em_Line_{}_Hist.png'.format(dir_fig,obs,round(peak_ref_wls[i])), format='png')
        fignum += 1
    results = np.array(results)
    return results, fignum

def line_stability(results, order, plot, fignum):
    """
    Create plot to compare emission line by emission line stability.
    """
    plt.figure(fignum, (10,8))
    plt.scatter(results[:,0],results[:,1],zorder=3)
    plt.grid()
    plt.ylabel('Std Dev ($\AA$)')
    plt.xlabel('Emission Line Peak ($\AA$)')
    plt.title('Emission Line Stability (Order {})'.format(order))
    if plot:
        plt.savefig('{}/Em_Line_Stability_{}.png'.format(dir_fig,order), format='png')
    fignum += 1
    return fignum

def save_results(results, obs, order):
    """
    Save results to a txt file
    """
    np.save('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/results/{}_results_{}'.format(obs,order), results)

#%%
"""
Access data
"""

obs = 'b0077'

data1 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data1.npy'.format(obs))
data2 = \
    np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data2.npy'.format(obs))

#%%
"""
Execute Main Funcions and Obtain Results (ORDER 1)
"""
fignum = 0
# List of interval parameters for each emission line to be modified per
# observation. (Usually 1.0 values work fine)
intervals_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#
# Get the reference emission line wavelengths & peak values
ref_wls_1, ref_pks_1, fignum = get_lines(data1[0][0],data1[0][1],1,True,fignum)

# Extract intervals of data around each emission line for each spectrum
data1_wls, data1_cts = get_line_arrays(data1, ref_wls_1, intervals_1)

# Remove NaN values from the intervals to prevent issues with fitting
data1_wls, data1_cts = remove_nans(data1_wls, data1_cts)

# Perform fitting of each emission line for each spectrum
line_fits_1, fignum = fitting(data1, data1_wls, data1_cts, ref_wls_1, intervals_1, False, fignum)

# Peform sigma clipping of data
line_fits_1 = data_clip(line_fits_1, 4, 4)

# Perform histogram fitting of results, extract results
results_1, fignum = histograms(line_fits_1, ref_wls_1, 1, 16, False, fignum)

# Create line by line uncertainty comparison plot
fignum = line_stability(results_1, 1, False, fignum)

# Save results to npy file
save_results(results_1,obs,1)

#%%
"""
Execute Main Funcions and Obtain Results (ORDER 2)
"""
fignum = 0
# List of interval parameters for each emission line to be modified per
# observation. (Usually 1.0 values work fine)
intervals_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Get the reference emission line wavelengths & peak values
ref_wls_2, ref_pks_2, fignum = get_lines(data2[0][0],data2[0][1],2,True,fignum)

# Extract intervals of data around each emission line for each spectrum
data2_wls, data2_cts = get_line_arrays(data2, ref_wls_2, intervals_2)

# Remove NaN values from the intervals to prevent issues with fitting
data2_wls, data2_cts = remove_nans(data2_wls, data2_cts)

# Perform fitting of each emission line for each spectrum
line_fits_2, fignum = fitting(data2, data2_wls, data2_cts, ref_wls_2, intervals_2, False, fignum)

# Peform sigma clipping of data
line_fits_2 = data_clip(line_fits_2, 4, 4)

# Perform histogram fitting of results, extract results
results_2, fignum = histograms(line_fits_2, ref_wls_2, 2, 8, True, fignum)

# Create line by line uncertainty comparison plot
fignum = line_stability(results_2, 2, True, fignum)

# Save results to npy file
save_results(results_2,obs,2)