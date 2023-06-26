from test3 import Bias, Flat, SciImage


def main():

    bias_instance = Bias()
    super_bias = bias_instance.make_superbias()

    flat_instance = Flat()
    super_flat = flat_instance.make_superflat()

    sci_img_instance = SciImage()
    sci_img = sci_img_instance.do_2d_reduction(input_format='ascii.no_header')


if __name__ == '__main__':

    main()







