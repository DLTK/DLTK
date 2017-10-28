# -*- coding: utf-8 -*-
"""Download and extract the IXI dataset 

url: http://brain-development.org/ixi-dataset/
ref: IXI – Information eXtraction from Images (EPSRC GR/S21533/02)

"""

import urllib
import os.path
import tarfile
import pandas as pd
import glob

EXTRACT_IMAGES = False
PROCESS_OTHER = True 

urls = {}
urls['t1'] = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'
urls['t2'] = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar'
urls['pd'] = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar'
urls['mra'] = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar'
urls['demographic'] = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'

fnames = {}
fnames['t1'] = 't1.tar'
fnames['t2'] = 't2.tar'
fnames['pd'] = 'pd.tar'
fnames['mra'] = 'mra.tar'
fnames['demographic'] = 'demographic.xls'


if EXTRACT_IMAGES:
    # Download all IXI data
    for key, url in urls.items():

        if not os.path.isfile(fnames[key]):
            print('Downloading {} from {}'.format(fnames[key], url))
            curr_file = urllib.FancyURLopener()
            curr_file.retrieve(url, fnames[key]) 
        else:
            print('File {} already exists. Skipping download.'.format(fnames[key]))

    # Extract the HH subset of IXI
    for key, fname in fnames.items():

        if (fname.endswith('.tar')):
            print('Extracting IXI HH data from {}.'.format(fnames[key]))

            if not os.path.exists(key):
                os.makedirs(key)

            t = tarfile.open(fname, 'r')
            for member in t.getmembers():
                if '-HH-' in member.name:
                    t.extract(member, key)   

    # Clean up .tar files
    for key, fname in fnames.items():
        if (fname.endswith('.tar')):
            os.remove(fname)

if PROCESS_OTHER:
    xls = pd.ExcelFile('demographic.xls')
    print(xls.sheet_names)
    
    df = xls.parse('Table')
    for index, row in df.iterrows():
        IXI_id = 'IXI{:03d}'.format(row['IXI_ID'])
        df.loc[index, 'IXI_ID'] = IXI_id
        
        t1_exists = len(glob.glob('./t1/{}*.nii.gz'.format(IXI_id)))
        t2_exists = len(glob.glob('./t2/{}*.nii.gz'.format(IXI_id)))
        pd_exists = len(glob.glob('./pd/{}*.nii.gz'.format(IXI_id)))
        mra_exists = len(glob.glob('./mra/{}*.nii.gz'.format(IXI_id)))
        
        if not t1_exists and not t2_exists and not pd_exists and not mra_exists:
            df.drop(index, inplace=True)
    
    df.to_csv('demographic_HH.csv', index=False)         