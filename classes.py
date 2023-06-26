import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import config
import app.pykosmos.kosmos as kosmos


class Bias:
    def __init__(self):
        self.input_directory = config.input_directory
        self.input_list_filename = config.input_file_list
        self.output_directory = config.output_directory
        self.output_filename = config.output_bias_filename

    def make_superbias(self, input_format='ascii.no_header', ax=None):
        assert config.output_bias_filename.endswith('hdf'), 'output_file must end in .hdf'

        bias_files = self._read_bias_files(input_format)
        bias_data = self._combine_bias_data(bias_files)

        self._save_bias_data(bias_data)
        self._plot_bias_data(bias_data, ax)

        return bias_data

    def _read_bias_files(self, input_format):
        bias_files = Table.read(os.path.join(self.input_directory, self.input_list_filename),
                                names=['impath'],
                                format=input_format)
        bias_files['impath'] = [os.path.join(self.input_directory, ifile) for ifile in bias_files['impath']]
        return bias_files

    @staticmethod
    def _combine_bias_data(bias_files):
        filtered_files = [file_path for file_path in bias_files['impath'] if
                          'bias' in os.path.basename(file_path).lower()]
        bias_data = kosmos.biascombine(filtered_files)
        return bias_data

    def _save_bias_data(self, bias_data):
        if self.output_directory is None:
            self.output_directory = config.output_directory
        with h5py.File(os.path.join(self.output_directory, config.output_bias_filename), 'w') as output_file:
            _ = output_file.create_dataset(name='bias', data=bias_data)

    def _plot_bias_data(self, bias_data, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        vmin, vmax = np.percentile(bias_data, (5, 98))
        im = ax.imshow(bias_data, origin='lower', aspect='auto', cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title('median bias frame {}'.format(config.output_bias_filename))
        plt.colorbar(mappable=im, ax=ax)
        plt.show(block=False)
        plt.draw()
        plt.savefig(os.path.join(self.output_directory, config.output_bias_filename.replace('hdf', 'pdf')))


class Flat:
    def __init__(self):
        self.input_directory = config.input_directory
        self.input_list_filename = config.input_file_list
        self.output_directory = config.output_directory
        self.output_filename = config.output_flat_filename

    def make_superflat(self, input_format='ascii.no_header', ax=None):
        assert config.output_flat_filename.endswith('hdf'), 'output_file must end in .hdf'

        flat_files = self.read_flat_files(input_format)
        flat_data = self._combine_flat_data(flat_files)

        self._save_flat_data(flat_data)
        self._plot_flat_data(flat_data, ax)

        return flat_data

    def read_flat_files(self, input_format):
        flat_files = Table.read(os.path.join(self.input_directory, self.input_list_filename),
                                names=['impath'],
                                format=input_format)
        flat_files['impath'] = [os.path.join(self.input_directory, ifile) for ifile in flat_files['impath']]
        return flat_files

    @staticmethod
    def _combine_flat_data(flat_files):
        filtered_files = [file_path for file_path in flat_files['impath'] if
                          'flat' in os.path.basename(file_path).lower()]
        flat_data = kosmos.flatcombine(filtered_files, bias=super_bias, illumcor=False)
        return flat_data

    def _save_flat_data(self, flat_data):
        if self.output_directory is None:
            self.output_directory = config.output_directory
        with h5py.File(os.path.join(self.output_directory, config.output_flat_filename), 'w') as output_file:
            _ = output_file.create_dataset(name='flat', data=flat_data)

    def _plot_flat_data(self, flat_data, ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        im = ax.imshow(flat_data, origin='lower', aspect='auto', cmap="inferno", vmin=0.9, vmax=1.1)
        ax.set_title('median flat frame, bias & response corrected {}'.format(config.output_flat_filename))
        plt.colorbar(mappable=im, ax=ax)
        plt.show(block=False)
        plt.draw()
        plt.savefig(os.path.join(self.output_directory, config.output_flat_filename.replace('hdf', 'pdf')))


class SciImage:
    def __init__(self):
        self.input_directory = config.input_directory
        self.input_list_filename = config.input_file_list
        self.output_directory = config.output_directory
        self.output_filename = config.output_target_filename

    def do_2d_reduction(self, input_format='ascii.no_header', illum=None, trim=True):

        target_files = self._read_target_files(input_format)
        sciimgs = self._process_sciimages(target_files, illum, trim)
        arcimgs = self._process_arcimages(target_files, trim)
        stdimgs = self._process_stdimages(target_files, trim)

        return sciimgs, arcimgs, stdimgs

    def _read_target_files(self, input_format):
        target_files = Table.read(os.path.join(config.input_directory, config.input_file_list),
                                  names=['impath'],
                                  format=input_format)
        target_files['impath'] = [os.path.join(self.input_directory, ifile) for ifile in target_files['impath']]
        return target_files

    def _process_sciimages(self, target_files, illum, trim):
        sciimgs = []
        filtered_target_files = [file_path for file_path in target_files['impath'] if
                                 'object' in os.path.basename(file_path).lower()]
        for impath in filtered_target_files:
            sciimg = kosmos.proc(impath, bias=super_bias, flat=super_flat, ilum=illum, trim=trim)
            output_filename = os.path.basename(impath).replace('.fits', '_flt.hdf')
            output_path = os.path.join(self.output_directory, output_filename)
            self._save_image_data(output_path, 'sciimage', sciimg)

            fig, ax = self._plot_image_data(sciimg, config.output_target_filename)
            self._save_plot(fig, ax, impath, '.pdf')

            sciimgs.append(sciimg)
        return sciimgs

    def _process_arcimages(self, target_files, trim):
        arcimgs = []
        filtered_arc_files = [file_path for file_path in target_files['impath'] if
                              'arc' in os.path.basename(file_path).lower()]
        for arc_file in filtered_arc_files:
            arcimg = kosmos.proc(arc_file, bias=super_bias, flat=super_flat, trim=trim)
            output_filename = os.path.basename(arc_file).replace('.fits', '_flt.hdf')
            output_path = os.path.join(self.output_directory, output_filename)
            self._save_image_data(output_path, 'arcimage', arcimg)

            fig, ax = self._plot_image_data(arcimg, config.output_arc_filename)
            self._save_plot(fig, ax, arc_file, '.pdf')

            arcimgs.append(arcimg)
        return arcimgs

    def _process_stdimages(self, target_files, trim):
        stdimgs = []
        filtered_std_files = [file_path for file_path in target_files['impath'] if
                              'std' in os.path.basename(file_path).lower()]
        for std_file in filtered_std_files:
            stdimg = kosmos.proc(std_file, bias=super_bias, flat=super_flat, ilum=None, trim=trim)
            output_filename = os.path.basename(std_file).replace('.fits', '_flt.hdf')
            output_path = os.path.join(self.output_directory, output_filename)
            self._save_image_data(output_path, 'stdimage', stdimg)

            fig, ax = self._plot_image_data(stdimg, config.output_std_filename)
            self._save_plot(fig, ax, std_file, '.pdf')

            stdimgs.append(stdimg)
        return stdimgs

    @staticmethod
    def _save_image_data(output_path, dataset_name, data):
        with h5py.File(output_path, 'w') as output_file:
            output_file.create_dataset(name=dataset_name, data=data)

    @staticmethod
    def _plot_image_data(data, output_filename):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        vmin, vmax = np.percentile(data, (5, 98))
        im = ax.imshow(data, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        corrections = 'bias and flat'
        ax.set_title(f'science frame, {corrections} corrected {output_filename}')
        plt.colorbar(mappable=im, ax=ax)
        return fig, ax

    def _save_plot(self, fig, ax, input_file, extension):
        output_filename = os.path.basename(input_file).replace('.fits', extension)
        output_path = os.path.join(self.output_directory, output_filename)
        plt.savefig(output_path)
        plt.close()


# bias_instance = Bias()
# super_bias = bias_instance.make_superbias()

# flat_instance = Flat()
# super_flat = flat_instance.make_superflat()

# sci_img_instance = SciImage()
# sci_img = sci_img_instance.do_2d_reduction()
