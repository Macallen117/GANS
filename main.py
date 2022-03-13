# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import keras
import tensorflow as tf


def print_info(name):
    print('Python: {}'.format(name))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_info(keras.__version__)
    print_info(tf.__version__)
