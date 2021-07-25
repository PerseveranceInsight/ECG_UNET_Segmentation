import numpy as np
import glob as gb
import sys
import os

class ECG_SEG_Dataset():
    def __init__(self, dataset_path):
        import wfdb
        from wfdb import processing
        leads_suffix = {0: '.i',
                        1: '.ii',
                        2: '.iii',
                        3: '.avr',
                        4: '.avl',
                        5: '.avf',
                        6: '.v1',
                        7: '.v2',
                        8: '.v3',
                        9: '.v4',
                        10: '.v5',
                        11: '.v6'}

