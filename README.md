# Mount Sinai Health Hackathon EKG Imaging Project

## Team Members:
Rryana Banashefski  
Cole McCollum  
Jackie Kim  
Matthew Epland  
Rahul Puppala  
Stephane Junior Nouafo Wanko  
Subrat Das  
Yurim Heo  
Jiang Yu

## Summary
Utilized the Resnet 101 model with transfer learning to classify images of 12 lead EKG traces.
ECG training data was obtained from [physionet](https://physionet.org/content/ptbdb/1.0.0/), and loaded with [wfdb](https://github.com/MIT-LCP/wfdb-python).  

### Results
The best results, obtained after fine tuning the final layer for 20 epochs, are:  

Train Loss: 0.0051, Acc: 0.9996  
Val Loss: 0.2338, Acc: 0.9260  

Training complete in 67m 57s  
Best Val Acc: 0.926020  

Confusion Matrix:  
| 133 |   7 |   0 |   0 |   4 |
|   6 | 146 |   0 |   5 |   2 |
|   0 |   1 | 158 |   0 |   0 |
|   3 |  11 |   0 | 143 |   2 |
|   3 |   3 |   0 |   0 | 153 |
 
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
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software. If you do not wish to use a virtual environment you can just run the first cell of the notebook - useful when running on cloud-based virtual machines.
```bash
python -m venv ~/.venvs/hackathon
source ~/.venvs/hackathon/bin/activate
pip install -r requirements.txt
```

## Running the Preprocessing Notebook

```bash
jupyter lab preprocessing/waveform_preproc.ipynb
```
