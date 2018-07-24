# LoLa

## Get the code

```
cd  CMSSW_9_4_6_patch1/src
cmsenv
git clone https://github.com/thaarres/LoLa.git
```

## Get the inputs

### The root files:
A script to prduce the necessary root files is here https://github.com/thaarres/VTaggingStudies/blob/W/CMSSW_9_4_6_patch1/src/SubstructureStudies/produceLoLa_W.py
You need E, px,py,pz of the 100 leading pT jet constituents, one signal and one background file. The script above should make it clear.

```
cd LoLa/preprocess
python PrepareInputsW.py
python ReorderFileW.py 
```
The first script, PrepareInputsW.py, needs your input files and converts root files to arrays stored in .h5 format. The second script reshuffles the content so that background and signal events are properly mixed.

## To Train
```
cd ../trainW/
python train.py 
```
