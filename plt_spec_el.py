"""
Modified version of plt_spec
"""

exec(open('utils_song.py').read())

extract_method = 'fox'
dir_spec = '/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/20190809/Reduce/r-side'
dir_fig = '/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/fig/20190809/'
std_obs = np.array(['r0051'])

datasets1 = []
datasets2 = []

fname_all = []
for file in os.listdir(dir_spec):
    if file.endswith('%s_specs.fits'%extract_method):
        fname_all.append(file)
fname_all.sort()
N_all = len(fname_all)

for i_file in range(N_all):
    #i_file  = 0
    shoe_temp = fname_all[i_file][0]
    file_temp = fname_all[i_file][0:5]
    fnum_temp = fname_all[i_file][1:5]
    dir_fig_temp = '%s/%s_cr'%(dir_fig,fnum_temp)

    if np.sum(std_obs==file_temp)!=1: continue

    print("Plotting %s ..."%file_temp)

    if os.path.isdir(dir_fig_temp)==False:
        os.mkdir(dir_fig_temp)

    data_temp = fits.open('%s/%s'%(dir_spec,fname_all[i_file]))[0].data

    for i_fiber in range(64):
        #i_fiber = 0
        print(i_fiber)
        mask_o1 = data_temp[0,:,0,i_fiber]>0.
        mask_o2 = data_temp[0,:,1,i_fiber]>0.
        
        datasets1.append([data_temp[0,:,0,i_fiber][mask_o1], data_temp[1,:,0,i_fiber][mask_o1]])
        datasets2.append([data_temp[0,:,1,i_fiber][mask_o2], data_temp[1,:,1,i_fiber][mask_o2]])
        
        ####
        fig = plt.figure(0, figsize=(24,16))
        fig.clf()

        ax = fig.add_subplot(211)
        #ax.set_xlim([8350,8800])
        ax.set_xlim([8360,8590])
        #ax.set_ylim([6,-1])
        ax.plot(data_temp[0,:,0,i_fiber][mask_o1], data_temp[1,:,0,i_fiber][mask_o1], 'r-', label='%s-%s%02d-O1'%(file_temp,shoe_temp,i_fiber+1))
        #ax.plot(data_temp[0,:,0,i_fiber][mask_o1], data_temp[2,:,0,i_fiber][mask_o1], 'k-', alpha=0.5) #, label='%s-%s%02d-O1'%(file_temp,shoe_temp,i_fiber+1))
        ax.legend(loc='upper left')
        ax.set_ylabel(r'Counts', fontsize=22)

        ax = fig.add_subplot(212)
        ax.set_xlim([8560,8790])
        ax.plot(data_temp[0,:,1,i_fiber][mask_o2], data_temp[1,:,1,i_fiber][mask_o2], 'b-', label='%s-%s%02d-O2'%(file_temp,shoe_temp,i_fiber+1))
        #ax.plot(data_temp[0,:,1,i_fiber][mask_o2], data_temp[2,:,1,i_fiber][mask_o2], 'k-', alpha=0.5) #, label='%s-%s%02d-O2'%(file_temp,shoe_temp,i_fiber+1))
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$\lambda\ {\rm (\AA)}$', fontsize=22)
        ax.set_ylabel(r'Counts', fontsize=22)

        fig.set_tight_layout(True)
        #fig.show()     # Not working
        fig.savefig('%s/%s_%02d_br_%s.pdf'%(dir_fig_temp,file_temp,i_fiber+1,extract_method), format='pdf', transparent=True)
        
    print("Done.")
    
#%%
"""
Save data
"""
datasets1, datasets2 = np.array(datasets1), np.array(datasets2)
for obs in std_obs:
    np.save('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data1'.format(obs), datasets1)
    np.save('/Users/Julian/Desktop/UofT/Internship/M2FS_Binary/m2fs_code/m2fs_plt/numpy_data/{}_data2'.format(obs), datasets2)