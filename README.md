# RKHS_Stochastic_Traj_Opt
Repository associated with our IROS 2023 submission. The repo contains trajectory optimizers for planning under multi-modal uncertainty.

https://user-images.githubusercontent.com/28586183/224532357-d03872ac-7825-48c6-80fb-3043d68ecf01.mp4

## Getting Started

1. Clone this repository:
```
git clone https://github.com/arunkumar-singh/RKHS_Stochastic_Traj_Opt.git
cd RKHS_Stochastic_Traj_Opt
```
2. Create a conda environment and install the dependencies:

```
conda create -n rkhs python=3.8
conda activate rkhs
pip install -r requirements.txt
```
### 1. Reduced Set Selection
``` 
params: 
Since we sample y coordinate using a discrete distribution, ydes_1 and ydes_2 are the two choices for the desired y-coordinates of the obstacle. 
ydes_1 - desired y-coordinate
ydes_2 - desired y-coordinate
prob - probability associated with ydes_1
```

1. 
``` 
cd IROS_repo_reduced 
```

2. 
```
python main_reduced.py --ydes_1 ${float} --ydes_2 ${float} --prob {float between 0 and 1}

```

The output of the above python code is a ```.mp4``` file stored in the folder ```IROS_repo_reduced/videos/```.The video shows an animation of the reduced set sample selection and the associated cross-entropy cost.
