#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:23:03 2022

@author: gliu
"""

def foo(fname):
    print(fname)
    
def foo1(fname="Default",arg2=None):
    print(fname)
    
fname2="Name"

foo(fname2)

foo1(fname2)

for n in np.arange(1,25,1):
    fname = "WHOI35_test_data_hamming60_24hrs_%i.mat" % (n) 
    
    # Calculation to get latt/lonn from file
    
    # runall your functions
    get_Data(fname)
    
    
    np.mean()
    
    
    print(fname)
    
    
    
    
a="ehy",b="eys"