"""
Julian's Notes:
    o The plots generated by the first block where all the analysis always
      seem to overlap and jumble up together. I think this can be solved by
      using plt.clf() in the functions.
"""
exec(open('elp_funcs.py').read())

#%%
"""
Execute ELP code for set of observations
"""
obs_list = ['b0073','b0074','b0076','b0077']

fignum = 0
for obs in obs_list:
    # Load data files
    data1 = \
        np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data1.npy'.format(obs))
    data2 = \
        np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data2.npy'.format(obs))
    
    # Run ELP functions
        # Order 1
    intervals_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ref_wls_1, ref_pks_1, fignum = get_lines(data1[0][0],data1[0][1],1,False,fignum)
    data1_wls, data1_cts = get_line_arrays(data1, ref_wls_1, intervals_1)
    data1_wls, data1_cts = remove_nans(data1_wls, data1_cts)
    line_fits_1, fignum = fitting(data1, data1_wls, data1_cts, ref_wls_1, intervals_1, False, fignum)
    line_fits_1 = data_clip(line_fits_1, 4, 4)
    results_1, fignum = histograms(line_fits_1, ref_wls_1, 1, 16, False, fignum)
    fignum = line_stability(results_1, 1, False, fignum)
    save_results(results_1,obs,1)
    
        # Order 2
    intervals_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ref_wls_2, ref_pks_2, fignum = get_lines(data2[0][0],data2[0][1],2,False,fignum)
    data2_wls, data2_cts = get_line_arrays(data2, ref_wls_2, intervals_2)
    data2_wls, data2_cts = remove_nans(data2_wls, data2_cts)
    line_fits_2, fignum = fitting(data2, data2_wls, data2_cts, ref_wls_2, intervals_2, False, fignum)
    line_fits_2 = data_clip(line_fits_2, 4, 4)
    results_2, fignum = histograms(line_fits_2, ref_wls_2, 2, 8, False, fignum)
    fignum = line_stability(results_2, 2, False, fignum)
    save_results(results_2,obs,2)

#%%
"""
Plot multi-observation comparison
"""
multi_results_1 = []
multi_results_2 = []
for obs in obs_list:
    multi_results_1.append(np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/results/{}_results_{}.npy'.format(obs,1)))
    multi_results_2.append(np.load('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/results/{}_results_{}.npy'.format(obs,2)))

x = []
for i in range(len(obs_list)):
    x.append(i)

fignum = 1000
for k in range(len(multi_results_1[0])):
    plt.figure(fignum,figsize=(8,6))
    plt.xticks(x,obs_list)
    plt.grid()
    for i in range(len(multi_results_1)):
        plt.scatter(i, multi_results_1[i][k][0], c='b', s=4, zorder=3)
    plt.xlabel('Observation')
    plt.ylabel('Wavelength ($\AA$)')
    plt.title('Multi-obs Emission Line Wavelength Drift ({}, Order {})\n'.format(get_obj(obs_list[0],obj_dict),1))
    
    fignum += 1

for k in range(len(multi_results_2[0])):
    plt.figure(fignum,figsize=(8,6))
    plt.xticks(x,obs_list)
    plt.grid()
    for i in range(len(multi_results_2)):
        plt.scatter(i, multi_results_2[i][k][0], c='b', s=4, zorder=3)
    plt.xlabel('Observation')
    plt.ylabel('Wavelength ($\AA$)')
    plt.title('Multi-obs Emission Line Wavelength Drift ({}, Order {})\n'.format(get_obj(obs_list[0],obj_dict),2))
    
    fignum += 1