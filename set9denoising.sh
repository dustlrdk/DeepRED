#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
python denoising.py 'image_F16_512rgb'
python denoising.py 'image_Lena512rgb'
python denoising.py 'kodim12'
python denoising.py 'kodim03'
python denoising.py 'image_House256rgb'
python denoising.py 'kodim02'
python denoising.py 'kodim01'
python denoising.py 'image_Peppers512rgb'
python denoising.py 'image_Baboon512rgb'