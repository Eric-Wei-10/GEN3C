# Frontier Generation
We keep track of the usage of all scripts and code here: 

version date: Dec. 25. 2025
## gen3c_frontier.py

This is the core script for the frontier generation pipeline. Here, the GEN3C model is loaded to GPU once, ran for multiple inference rounds to generate videos of multiple **scenes**, **trajectories**, and **seeds**. 

To run this script, please directly run from root:

```
bash generate_frontiers.sh
```

### Important parameters

```
--num_video_frames 
# default 121 for the best performance, although ok to set to any value n such that n % 8 == 1

--num_steps 
# defaults to 10. We observe little improvements on the video quality with higher steps.

--agent_speed 
# defaults to 0.5 m/s. No faster than 0.5 due to robot limitation

--sample_angle 
# defaults to 15 degrees. The angle increment value for sweeping the yaw direction (theta). Typically between 0 and 30 degrees.

--num_seeds 
# defaults to 4. The number of seeds, also the number of videos, generated for each trajectory. The seeds are generated with a torch random engine

--height
# defaults to 704. The resolution of generated videos, finetune this parameter to balance with quality and GPU memory usage.

--width
# defaults to 1280. The resolution of generated videos, finetune this parameter to balance with quality and GPU memory usage.
```