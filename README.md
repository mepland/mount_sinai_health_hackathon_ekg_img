# Mount Sinai Health Hackathon EKG Imaging Project

## Team Members:
Bryana Banashefski  
Cole McCollum  
Jackie Kim  
Matthew Epland  
Rahul Puppala  
Stephane Junior Nouafo Wanko  
Subrat Das  
Yurim Heo  
Jiang Yu

## Summary
Transfer learning with Resnet 101 model is used in the projects. 
Best results we achieved after the model was fine trained for 20 epoches: <br/>
----------
train Loss: 0.0051 Acc: 0.9996
val Loss: 0.2338 Acc: 0.9260 <br/>

Training complete in 67m 57s
Best val Acc: 0.926020 <br/>

Confusion Matrix: <br/>
 [[133   7   0   0   4]<br/>
 [  6 146   0   5   2]<br/>
 [  0   1 158   0   0]<br/>
 [  3  11   0 143   2]<br/>
 [  3   3   0   0 153]] <br/>
 

Open ECG data from [physionet](https://physionet.org/content/ptbdb/1.0.0/), loaded with [wfdb](https://github.com/MIT-LCP/wfdb-python)  

## Cloning the Repository
ssh  
```bash
git clone git@github.com:mepland/mount_sinai_health_hackathon_ekg_img.git
```

https  
```bash
git clone https://github.com/mepland/mount_sinai_health_hackathon_ekg_img.git
```

## Installing Dependencies
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software. If you do not wish to use a virtual environment you can just run the first few cells of the notebook - useful when running on cloud-based virtual machines.
```bash
python -m venv ~/.venvs/hackathon
source ~/.venvs/hackathon/bin/activate
pip install -r requirements.txt
```

## Running the Notebooks

```bash
jupyter lab preprocessing/waveform_preproc.ipynb
```
