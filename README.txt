Use plt_spec_el.py (modified version of plt_spec) to save the spectrum data as NPY files.
	- Each observation produces 2 files, one for each order.
	- Only change from plt_spec to plt_spec_el is added variables that store spectrum data during the plotting loop
	  and then saves the data in NPY files after the plotting loop is completed.
		- You will need to change the directory of where the files are saved for your device
	- Requires utils_song.py

spectra_2d_plt.py:
	- The visualization code of the spectrum data.
	- Directory to access the data must be changed for your device.
	- The code uses cells to separate plotting options of contourf and imshow.
	- This code is manual and is purely for visual analysis.

obj_dict.py:
	- A file that stores a dictionary relating observations to objects, as well as a function to access them.
	- If you wish to continue using it you can update the dictionary as necessary, but the function will not raise an
	  error if the observation key is not in the dictionary.
	- For use only for plots in other code.

emission_lines_plt_auto.py:
	- Automatic version of the emission_lines_plt.py
	- Change the directories for your device.
	- The code has cells for manual tweaking/testing, but can be run entirely without stopping
	- Does only 1 observation at a time
	- Recommend messing around with emission_lines_plt (the manual version) to get a good understanding of how the
	  code works and the different types of issues it can run into if not used properly.
	- The code outputs two saved txt files of the results of the histograms, one for each order.
	- All functions take 'fignum' as an input, and output fignum. This is to create multiple separate plots.
	- The functions:
		- get_lines(): Identifies set of emission lines (wavelengths & cts) for use as a reference to detect
			       emission lines for all spectra of the observation
			- The way this function identifies the set of reference wavelengths is by referencing a pre-defined
			  array of wavelengths (integers) and finding them in the first spectrum.
				- This works only if the emission line wavelengths do not border on integers and the drift
				  between spectra is sufficently small such that the emission lines across the spectra all 
				  have the same integer values and differ only by the decimal values.
				- This is not a great method. I have had no issues with it so far, but there is no
				  guarantee it will work for all observations, present and future.
		- get_line_arrays(): Obtains wavelength arrays of given interval around each referenced emission line
				     in all spectra of the observation
		- remove_nans(): Quality control function. Removes NaN valued data points from wavelength intervals
				 obtained from get_line_arrays.
		- fitting(): Performs gaussian fitting on each emission line of each spectrum in the observation. Outputs
			     the results of the fittings.
		- data_clip(): Performs sigma clipping on results from fitting().
		- histograms_old(): Creates histograms of results, fits gaussian to the histograms.
			- Not recommended to use, does not have parameter for binsize and gaussian fitting of histograms
			  was deemed unnecessary.
		- histograms(): Creates histograms of results (with modifiable binsize), generates gaussian from
				data by calculating mean & stddev and plots this gaussian on top of histogram.
			- This gaussian is not fitted to the histogram. It is just calculated from the data and plotted.
			- The amplitude of the plotted gaussian is the count value of the tallest bin.
		- line_stability(): Plots a comparison of emission line by emission line stability.
		- save_results(): Saves results of histograms() to a txt file.
			- This function references a directory that must be changed to your device
			- This saved data is to be used to compare observation-to-observation stability & wavelength drift.

elp_funcs.py:
	- File that has only the functions from emission_lines_plt_auto.py
	- Directories will need to be changed to your device
	- Used only to import the functions to multi_obs_ELP.py

multi_obs_ELP.py:
	- Multi-observation version of emission_lines_plt_auto.py
	- Directories will need to be changed to your device
	- First runs the automatic code for each observation, then errorbar plots emission lines means & stddevs to show
	  observation-to-observation wavelength drift.
	- Can run into an error:
		- get_lines() is not perfectly stable for multi-observation use. On the occasion that one of the emission
		  lines you choose to analyze is a double-line the function will sometimes pick up both peaks, and sometimes
		  only pick up one. In the first case this causes an error and the code will stop.
			- Can be fixed by manually changing the reference wavelength array in elp_funcs by adding decimals
			  and forcing the function to choose the closest peak or by removing/changing the emission line 
			  you want to analyze.
	- The plots generated by the first set of functions to get the fitting results and histogram analysis completely
	  overlap into a mess every time.
		- I believe implementation of plt.clf() in elp_funcs.py may solve this issue. Alternatively can just ignore
		  these plots.
		- These plots do not interfere with the multi-observation comparison plots.