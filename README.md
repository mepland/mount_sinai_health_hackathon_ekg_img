# Mount Sinai Health Hackathon EKG Imaging Project

# 2023 Paper
Following the [October 2019 weekend hackathon](https://events.mountsinaihealth.org/event/healthhackathon),
Subrat Das and Matthew Epland resumed work on this project in May 2020.
The model was switched to an austere variation of MobileNetV3,
designed to be small enough to fit in the memory of a RTX 2080Ti GPU available to the authors.
After a lengthy publication process,
this work was finally published in April 2023 as
"**[Interpretation of EKG with Image Recognition and Convolutional Neural Networks](https://doi.org/10.1016/j.cpcardiol.2023.101744)**"
in Current Problems in Cardiology,
with assistance from Jiang Yu
and supervision by Professor Ranjit Suri.

## Paper Authors
- Subrat Das, MD
- Matthew Epland, PhD
- Jiang Yu, PhD
- Ranjit Suri, MD

## Paper Abstract
Electrocardiograms (EKG) form the backbone of all cardiovascular diagnosis, treatment and follow up. Given the pivotal role it plays in modern medicine, there have been multiple efforts to computerize the EKG interpretation with algorithms to improve efficiency and accuracy. Unfortunately, many of these algorithms are machine specific and run-on proprietary signals generated by that machine, hence not generalizable. We propose the development of an image recognition model which can be used to read standard EKG strips. A convolutional neural network (CNN) was trained to classify 12-lead EKGs between seven clinically important diagnostic classes. An austere variation of the MobileNetV3 model was trained from the ground up on publicly available labeled training set. The precision per class varies from 52-91%. This is a novel approach to EKG interpretation as an image recognition problem.

## Paper Model Files
The trained MobileNetV3 model and associated performance metrics can be found in the
[best_model_for_paper/2020-07-16_mobilenetv3_small_custom_800_best](best_model_for_paper/2020-07-16_mobilenetv3_small_custom_800_best)
directory.

# 2019 Hackathon

## Hackathon Team Members
- Cole McCollum
- Jackie Kim
- Jiang Yu
- Matthew Epland
- Rahul Puppala
- Rryana Banashefski
- Stephane Junior Nouafo Wanko
- Subrat Das
- Yurim Heo

## Hackathon Summary
Utilized the Resnet 101 model with transfer learning to classify images of 12 lead EKG traces.
ECG training data was obtained from [physionet](https://physionet.org/content/ptbdb/1.0.0/),
and loaded with [wfdb](https://github.com/MIT-LCP/wfdb-python).

### Results
The best results, obtained after fine tuning the final layer for 20 epochs, are:

```
Train Loss: 0.0051, Acc: 0.9996
Val Loss: 0.2338, Acc: 0.9260

Training complete in 67m 57s
Best Val Acc: 0.926020
```

Confusion Matrix:
```
[[133,   7,   0,   0,   4]
 [  6, 146,   0,   5,   2]
 [  0,   1, 158,   0,   0]
 [  3,  11,   0, 143,   2]
 [  3,   3,   0,   0, 153]]
```

## Running the Preprocessing Script
Pick j based on the CPU cores you have available, see `--help` for details.
```bash
python -u preprocessing.py -j 22
```

## Counting Output Images
Returns sorted values in CSV format, pipe directly to a file with `> counts.csv` at the end if desired.
```bash
cd preprocessing/output/im_res_800

find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
	printf "\"$dir\"|" | sed -r 's/\.\///g'
	find "$dir" -type f | wc -l
done | sort -t '|' -nrk2 | sed -r 's/\|/,/g'
```

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
