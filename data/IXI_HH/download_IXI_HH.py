# -*- coding: utf-8 -*-
"""Download and extract the IXI Hammersmith Hospital 3T dataset

url: http://brain-development.org/ixi-dataset/
ref: IXI – Information eXtraction from Images (EPSRC GR/S21533/02)

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.standard_library import install_aliases  # py 2/3 compatability
install_aliases()

from urllib.request import FancyURLopener

import os.path
import tarfile
import pandas as pd
import glob
import SimpleITK as sitk
import numpy as np

DOWNLOAD_IMAGES = True
EXTRACT_IMAGES = True
PROCESS_OTHER = True
RESAMPLE_IMAGES = True
CLEAN_UP = True


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def reslice_image(itk_image, itk_ref, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(itk_ref)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


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


if DOWNLOAD_IMAGES:
    # Download all IXI data
    for key, url in urls.items():

        if not os.path.isfile(fnames[key]):
            print('Downloading {} from {}'.format(fnames[key], url))
            curr_file = FancyURLopener()
            curr_file.retrieve(url, fnames[key])
        else:
            print('File {} already exists. Skipping download.'.format(
                fnames[key]))

if EXTRACT_IMAGES:
    # Extract the HH subset of IXI
    for key, fname in fnames.items():

        if (fname.endswith('.tar')):
            print('Extracting IXI HH data from {}.'.format(fnames[key]))
            output_dir = os.path.join('./orig/', key)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            t = tarfile.open(fname, 'r')
            for member in t.getmembers():
                if '-HH-' in member.name:
                    t.extract(member, output_dir)


if PROCESS_OTHER:
    # Process the demographic xls data and save to csv
    xls = pd.ExcelFile('demographic.xls')
    print(xls.sheet_names)

    df = xls.parse('Table')
    for index, row in df.iterrows():
        IXI_id = 'IXI{:03d}'.format(row['IXI_ID'])
        df.loc[index, 'IXI_ID'] = IXI_id

        t1_exists = len(glob.glob('./orig/t1/{}*.nii.gz'.format(IXI_id)))
        t2_exists = len(glob.glob('./orig/t2/{}*.nii.gz'.format(IXI_id)))
        pd_exists = len(glob.glob('./orig/pd/{}*.nii.gz'.format(IXI_id)))
        mra_exists = len(glob.glob('./orig/mra/{}*.nii.gz'.format(IXI_id)))

        # Check if each entry is complete and drop if not
        # if not t1_exists and not t2_exists and not pd_exists and not mra
        # exists:
        if not (t1_exists and t2_exists and pd_exists and mra_exists):
            df.drop(index, inplace=True)

    # Write to csv file
    df.to_csv('demographic_HH.csv', index=False)

if RESAMPLE_IMAGES:
    # Resample the IXI HH T2 images to 1mm isotropic and reslice all
    # others to it
    df = pd.read_csv('demographic_HH.csv', dtype=object, keep_default_na=False,
                     na_values=[]).as_matrix()

    for i in df:
        IXI_id = i[0]
        print('Resampling {}'.format(IXI_id))

        t1_fn = glob.glob('./orig/t1/{}*.nii.gz'.format(IXI_id))[0]
        t2_fn = glob.glob('./orig/t2/{}*.nii.gz'.format(IXI_id))[0]
        pd_fn = glob.glob('./orig/pd/{}*.nii.gz'.format(IXI_id))[0]
        mra_fn = glob.glob('./orig/mra/{}*.nii.gz'.format(IXI_id))[0]

        t1 = sitk.ReadImage(t1_fn)
        t2 = sitk.ReadImage(t2_fn)
        pd = sitk.ReadImage(pd_fn)
        mra = sitk.ReadImage(mra_fn)

        # Resample to 1mm isotropic resolution
        t2_1mm = resample_image(t2)
        t1_1mm = reslice_image(t1, t2_1mm)
        pd_1mm = reslice_image(pd, t2_1mm)
        mra_1mm = reslice_image(mra, t2_1mm)

        output_dir = os.path.join('./1mm/', IXI_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('T1: {} {}'.format(t1_1mm.GetSize(), t1_1mm.GetSpacing()))
        print('T2: {} {}'.format(t2_1mm.GetSize(), t2_1mm.GetSpacing()))
        print('PD: {} {}'.format(pd_1mm.GetSize(), pd_1mm.GetSpacing()))
        print('MRA: {} {}'.format(mra_1mm.GetSize(), mra_1mm.GetSpacing()))

        sitk.WriteImage(t1_1mm, os.path.join(output_dir, 'T1_1mm.nii.gz'))
        sitk.WriteImage(t2_1mm, os.path.join(output_dir, 'T2_1mm.nii.gz'))
        sitk.WriteImage(pd_1mm, os.path.join(output_dir, 'PD_1mm.nii.gz'))
        sitk.WriteImage(mra_1mm, os.path.join(output_dir, 'MRA_1mm.nii.gz'))

        # Resample to 2mm isotropic resolution
        t2_2mm = resample_image(t2, out_spacing=[2.0, 2.0, 2.0])
        t1_2mm = reslice_image(t1, t2_2mm)
        pd_2mm = reslice_image(pd, t2_2mm)
        mra_2mm = reslice_image(mra, t2_2mm)

        output_dir = os.path.join('./2mm/', IXI_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('T1: {} {}'.format(t2_2mm.GetSize(), t1_2mm.GetSpacing()))
        print('T2: {} {}'.format(t2_2mm.GetSize(), t2_2mm.GetSpacing()))
        print('PD: {} {}'.format(pd_2mm.GetSize(), pd_2mm.GetSpacing()))
        print('MRA: {} {}'.format(mra_2mm.GetSize(), mra_2mm.GetSpacing()))

        sitk.WriteImage(t1_2mm, os.path.join(output_dir, 'T1_2mm.nii.gz'))
        sitk.WriteImage(t2_2mm, os.path.join(output_dir, 'T2_2mm.nii.gz'))
        sitk.WriteImage(pd_2mm, os.path.join(output_dir, 'PD_2mm.nii.gz'))
        sitk.WriteImage(mra_2mm, os.path.join(output_dir, 'MRA_2mm.nii.gz'))


if CLEAN_UP:
    # Remove the .tar files
    for key, fname in fnames.items():
        if (fname.endswith('.tar')):
            os.remove(fname)

    # Remove all data in original resolution
    os.system('rm -rf orig')
