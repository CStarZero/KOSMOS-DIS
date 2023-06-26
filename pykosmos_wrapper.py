import os
import numpy as np
import app.pykosmos.kosmos as kosmos
import specutils
from matplotlib import pyplot as plt

import h5py
from astropy.table import Table
from astropy import units as u

import sys

sys.path.append('/Users/cstarzero/Documents/kosmos/data')


# TODO list
# Currently write HDF5 files, should it write fits files (e.g. for DS( quick view))
# Record header information in output
# Better way to do plotting where you can combine after the fact?
# Comment functions


def make_superbias(input_list_filename, output_filename='bias.hdf',
                   input_format='ascii.no_header', input_directory='./',
                   output_directory=None, interactive_plots=True, ax=None, ):
    """ Make a super bias. Output file and plots will be put in the output directory, default is the input_directory"""
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    bias_files = Table.read(os.path.join(input_directory, input_list_filename), names=['impath'], format=input_format)
    bias_files['impath'] = [os.path.join(input_directory, ifile) for ifile in bias_files['impath']]
    bias = kosmos.biascombine(bias_files['impath'], )
    # Save
    if output_directory is None:
        output_directory = input_directory
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='bias', data=bias)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    vmin, vmax = np.percentile(bias, (5, 98))
    im = ax.imshow(bias, origin='lower', aspect='auto', cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title('median bias frame {}'.format(output_filename))
    plt.colorbar(mappable=im, ax=ax)
    plt.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    plt.draw()
    if interactive_plots:
        input('Press enter to exit interactive')
    return bias


def make_superflat(input_list_filename, bias, output_filename='flat.hdf',
                   input_format='ascii.no_header', input_directory='./',
                   output_directory=None, interactive_plots=True, verbose=False, ax=None):
    """ Make a super flat. Output file and plots will be put in the output directory, default is the input_directory"""
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    flat_files = Table.read(os.path.join(input_directory, input_list_filename), names=['impath'], format=input_format)
    flat_files['impath'] = [os.path.join(input_directory, ifile) for ifile in flat_files['impath']]
    flat_output = kosmos.flatcombine(flat_files['impath'], bias=bias)
    if isinstance(flat_output, tuple):
        flat, illum = flat_output
    else:
        flat = flat_output
        illum = None
    # Save
    if output_directory is None:
        output_directory = input_directory
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='flat', data=flat)
        if illum is not None:
            illumset = output_file.create_dataset(name='illum', data=illum)
    if verbose:
        if illum is not None:
            print('illuminated shape {}'.format(illum.shape))
        print('flat shape {}'.format(flat.shape))
        print('flat unit {}'.format(flat.unit))
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    im = ax.imshow(flat, origin='lower', aspect='auto', cmap="inferno", vmin=0.9, vmax=1.1)
    ax.set_title('median flat frame, bias & response corrected {}'.format(output_filename))
    plt.colorbar(mappable=im, ax=ax)
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    if illum is not None:
        return flat, illum
    else:
        return flat


def do_2d_reduction(input_filename, bias, flat, illum=None, trim=True, input_directory='./',
                    output_filename=None, output_directory=None, interactive_plots=True, CR=True, verbose=False, ax=None,
                    ):
    """ Bias and Flat field a 2D spectrogram"""
    sciimg = kosmos.proc(os.path.join(input_directory, input_filename),
                         bias=bias, flat=flat, ilum=illum, trim=trim, )
    # Save
    if output_directory is None:
        output_directory = input_directory
    if output_filename is None:
        output_filename = input_filename.replace('.fits', '_flt.hdf')
    assert output_filename.endswith('hdf'), 'output_file ({}) must end in .hdf'.format(output_filename)

    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='sciimage', data=sciimg)
    if verbose:
        print('sci shape {}'.format(sciimg.shape))
        print('sci unit {}'.format(sciimg.unit))
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    vmin, vmax = np.percentile(sciimg, (5, 98))
    im = ax.imshow(sciimg, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    if illum is not None:
        corrections = 'bias, flat, and response'
    else:
        corrections = 'bias and flat'
    ax.set_title('science frame, {} corrected {}'.format(corrections, output_filename))
    plt.colorbar(mappable=im, ax=ax)
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    return sciimg


def trace_spectrum(image2d, output_filename, input_directory='./',
                   output_directory=None, interactive_plots=True, ):
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    if output_directory is None:
        output_directory = input_directory
    fig, ax = plt.subplots(1, 1)
    trace = kosmos.trace(image2d, display=True, ax=ax, )
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='trace', data=trace)
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    plt.close(fig)
    return trace


def extract_spectrum(image2d, trace, output_filename, input_directory='./',
                     output_directory=None, interactive_plots=True, arc=False, ):
    """
    Output file is a fits file
    """
    assert output_filename.endswith('fits'), 'output_file must end in .fits'
    if output_directory is None:
        output_directory = input_directory
    fig, ax = plt.subplots(1, 1)
    obj_spectrum, sky_spectrum = kosmos.BoxcarExtract(image2d, trace, display=True, ax=ax, )
    if 'x1d' not in output_filename:
        output_filename = output_filename.replace('.fits', '_x1d.fits')
    obj_spectrum.write(os.path.join(output_directory, output_filename), format='tabular-fits', overwrite=True)
    sky_spectrum.write(os.path.join(output_directory, output_filename.replace('.fits', '_sky.fits')),
                       format='tabular-fits', overwrite=True)
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '_extract.pdf')))
    fig_spectrum, ax_spectrum = plt.subplots(1, 1)
    if not arc:
        ax_spectrum.plot(obj_spectrum.spectral_axis.value, obj_spectrum.flux.value - sky_spectrum.flux.value)
    else:
        ax_spectrum.plot(obj_spectrum.spectral_axis.value, obj_spectrum.flux.value)
    ax_spectrum.set_xlabel(obj_spectrum.spectral_axis.unit)
    ax_spectrum.set_ylabel(obj_spectrum.flux.unit)
    ax_spectrum.set_title('Boxcar extraction')
    if interactive_plots:
        input('Press enter to exit interactive')
    fig_spectrum.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))
    plt.close(fig)
    plt.close(fig_spectrum)
    return obj_spectrum, sky_spectrum


def get_wavelength_solution(image, arc_spectrum, output_filename='disp.hdf', input_directory='./',
                            output_directory='./', blue_left=True, interactive_plots=True, ):
    if output_directory is None:
        output_directory = input_directory
    wapprox = (np.arange(image.shape[1]) - image.shape[1] / 2)
    if not blue_left:
        wapprox = wapprox[::-1]
    if 'DISPDW' in image.header.keys():
        wapprox = wapprox * image.header['DISPDW'] + image.header['DISPWC']
    wapprox = wapprox * u.angstrom
    kosmos_dir = os.path.dirname(kosmos.__file__)
    henear_tbl = Table.read(os.path.join(kosmos_dir, 'resources/linelists/apohenear.dat'),
                            names=['wave', 'name'], format='ascii')
    henear_tbl['wave'].unit = u.angstrom
    apo_henear = henear_tbl['wave']
    # Map pixel to wavelength
    xpts, wpts = kosmos.identify_nearest(arc_spectrum, wapprox=wapprox, linewave=apo_henear, )
    wl_soln_tbdata = Table([xpts, wpts], names=['pixel', 'wavelength'])
    wl_soln_tbdata.write(output_filename, format='ascii.csv', overwrite=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(wapprox.value, arc_spectrum.flux.value)
    ax.set_xlabel(wapprox.unit)
    ax.set_ylabel(arc_spectrum.flux.unit)
    ymin, ymax = ax.get_ylim()
    ax.vlines(wpts, color='r', ls=':', ymin=ymin, ymax=ymax)

    if interactive_plots:
        input('Press enter to exit interactive')
    fig.savefig(os.path.join(output_directory, output_filename.replace('.hdf', '.pdf')))
    plt.close(fig)
    return xpts, wpts


def calibrate_wavelengths(xpts, wpts, obj_spectrum, sky_spectrum, output_filename, input_directory='./',
                          output_directory='./', interactive_plots=True, ):
    if output_directory is None:
        output_directory = input_directory
    assert output_filename.endswith('.fits'), 'output_filename must end in .fits'
    if 'wx1d' not in output_filename:
        output_filename = output_filename.replace('.fits', '_wx1d.fits')

    # Create sky subtracted spectrum
    obj_flux = obj_spectrum.flux - sky_spectrum.flux
    sky_sub_obj_spectrum = specutils.Spectrum1D(flux=obj_flux, spectral_axis=obj_spectrum.spectral_axis,
                                                uncertainty=obj_spectrum.uncertainty)
    # Apply the wavelength solution to the sky subtracted spectrum
    obj_spectrum_wavelength_calibrated = kosmos.fit_wavelength(sky_sub_obj_spectrum, xpts, wpts, mode='interp', deg=3)
    obj_spectrum_wavelength_calibrated.write(os.path.join(output_directory, output_filename), format='tabular-fits',
                                             overwrite=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(obj_spectrum_wavelength_calibrated.spectral_axis, obj_spectrum_wavelength_calibrated.flux)
    ax.set_xlabel(obj_spectrum_wavelength_calibrated.spectral_axis.unit)
    ax.set_ylabel(obj_spectrum_wavelength_calibrated.flux.unit)
    ax.set_title('Wavelength Calibrated Spectrum')
    if interactive_plots:
        input('Press enter to exit interactive')
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))
    plt.close(fig)
    return obj_spectrum_wavelength_calibrated


def make_sens_func(std_filename, std_spectrum, output_filename='sens.fits', output_directory='./',
                   interactive_plots=True, mode='linear', ):
    assert output_filename.endswith('.fits'), 'output_filename must end in .fits'
    standardstar = kosmos.onedstd(std_filename)
    sens_func = kosmos.standard_sensfunc(std_spectrum, standardstar, mode=mode, )
    sens_func.write(os.path.join(output_directory, output_filename), format='tabular-fits', overwrite=True)

    fig, ax = plt.subplots(1, 1)
    ax.plot(sens_func.spectral_axis, sens_func.flux)
    ax.set_xlabel(sens_func.spectral_axis.unit)
    ax.set_ylabel('Sensitivity ({})'.format(sens_func.flux.unit))

    if interactive_plots:
        input('Press enter to exit interactive')
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))
    plt.close(fig)
    return sens_func


def apply_airmass_correction(header, spectrum, extinction_file='apoextinct.dat'):
    """
    Spectrum should be wavelength calibrated
    """
    # Get the airmass from the Headers... no fancy way to do this I guess? 
    zd = header['ZD'] / 180.0 * np.pi
    airmass = 1.0 / np.cos(zd)  # approximate Zenith Distance -> Airmass conversion
    # Select the observatory-specific airmass extinction profile from the provided "extinction" library
    Xfile = kosmos.obs_extinction(extinction_file)
    spectrum_airmass_corr = kosmos.airmass_cor(spectrum, airmass, Xfile)
    # TODO: this should write a file and maybe make a plot?
    return spectrum_airmass_corr


def flux_calibrate(obj_spectrum_wavelength_calibrated, sens_func, output_filename, output_directory='./',
                   interactive_plots=True, ):
    assert output_filename.endswith('.fits'), 'output_filename must end in .fits'
    flux_calib_spectrum = kosmos.apply_sensfunc(obj_spectrum_wavelength_calibrated, sens_func, )
    flux_calib_spectrum.write(os.path.join(output_directory, output_filename), format='tabular-fits', overwrite=True)
    fig, ax = plt.subplots(1, 1)
    ax.plot(flux_calib_spectrum.spectral_axis.value, flux_calib_spectrum.flux.value)
    ax.set_xlabel(flux_calib_spectrum.spectral_axis.unit)
    ax.set_ylabel(flux_calib_spectrum.flux.unit)
    ax.set_title('Flux Calibrated Spectrum')
    if interactive_plots:
        input('Press enter to exit interactive')
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))
    plt.close(fig)
    return flux_calib_spectrum
