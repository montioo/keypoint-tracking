# Training Overview:

## Visualization Script

```python3
python3 train_perf_plot.py <logfile_name>
```


## keypoint_training_2021-01-30_20:48:33.log

- Training data with view top down and camera orbiting
- Domain randomization for stick and table
- But stick always laid on the table
- 62000 train, 8000 validation
- Did 120 epochs

## keypoint_training_2021-02-01_17:53:08.log

- Additional training data with robot gripper in the image
- Continuing with model from 30.01.2021 epoch 118 (had 120 epoch but I forgot to save the last one :D )
- 88000 train, 12000 validation
- pixel accuracy continued to be good right away
- depth needed a bit of training

