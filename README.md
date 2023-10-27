# RKHS_Stochastic_Traj_Opt
Repository associated with our IROS 2023 (accepted) paper https://arxiv.org/abs/2310.08270. The repo contains trajectory optimizers for planning under multi-modal uncertainty.

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

### 2. Comparison with baseline [6]

**The following works on synthetically generated data but in general ,it is possible to use any dataset of your choice after certain cosmetic changes in the relevant parts of the code.**

Link to sample data corresponding to step 3: https://drive.google.com/drive/folders/1zKjVDz-YaB2eF5DejDLMigOodpF_T-2Z?usp=sharing

1. 
```
cd IROS_repo_comparison
```

2. To reproduce the comparison shown in the video we use the data stored in ```sample_data``` folder.
```
python visualize_sample_data_ours.py
python visualize_sample_data_baseline.py
```
The outputs of the above python codes are two ```.mp4``` files stored in ```IROS_repo_comparison/videos/```.The videos show animations of the planner output trajectories corresponding to our approach and that of [6].

3. It is possible to generate fresh synthetic data by running the script ```data_generation.py```
```
python data_generation.py
```

The script has certain modifiable parameters to generate data.Please see the comments in the script for further information. The output data is stored in the folder ```data_synthetic```.Each data file contains an obstacle configuration and collection of possible trajectories of the obstacle.

Once the data is generated , run:
```
python comparison.py
```
This will run the optimizers for [6] and our approach on all the data files in ```data_synthetic``` folder and the resulting output files,which we again call as data files,are stored in ```data_comparison``` folder. Each data file contains trajectory information of the ego vehicle corresponding to approach of [6] and our approach.

Now that we have the optimizer/planner output we can run comparisons between the [6] and our approach. We calculate the ```average number of collision-free trajectories``` for the two approaches for each data file in ```data_comparison``` folder. 
```
python stats_comparison.py
```
The output of the above script is ```stats_overall.txt``` which is stored in ```data_comparison``` folder. The file contains information about the performance of the two approaches in terms of ```average number of collision-free trajectories```.

Finally,to visualize the comparison in terms of ```box``` and ```violin``` plots run:
```
python plotting_box_plots.py
```



