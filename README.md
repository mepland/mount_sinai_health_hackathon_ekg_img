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

## Summary
TODO

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
