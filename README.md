# Ant_racer, 
### a virtual multi-agent pursuit-evasion platform based on OpenAI/Gym and Mujoco
This project is a variant of multi-agent game platform **TripleSumo ([Publication](https://link.springer.com/chapter/10.1007/978-3-031-15908-4_15), [Repository]( https://github.com/niart/triplesumo))**. 
A live demo of this game can be found in **[This Video](https://www.youtube.com/watch?v=egSRK1eWnf4)**. You're welcome to visit the **[author's Youtube page](https://www.youtube.com/@intelligentautonomoussyste5467/videos)** to find more about her work. Contact her at **niwang.cs@gmail.com** if you have inquiry.

Steps of installing Ant_racer:
1. Download [Mujoco200](https://www.roboti.us/download.html), rename the package into mujoco200, then extract it in 
   ```/home/your_username/.mujoco/ ```, then download the [license](https://www.roboti.us/license.html) into the same directory
2. Add ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/your_username/.mujoco/mujoco200/bin``` to your ```~/.bashrc```, and then ```source ~/.bashrc```
3. Use Anaconda to create a virtual environment 'ant_racer' with ```conda env create -f ant_racer_env.yml```; Then ```conda activate ant_racer```.
4. ```git clone https://github.com/niart/Ant_racer.git``` and ```cd Ant_racer```
5. Use the ```gym``` foler of this repository to replace the ```gym``` installed in your conda environment ant_racer. 
6. To test the demo, run ```python chase_demo.py```. If you meet error ```Creating window glfw ... ERROR: GLEW initalization error: Missing GL version```, you may add ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so``` to ```~/.bashrc```, then ```source ~/.bashrc```. 

A simple RL algorithm interface has been written in ```chase_runmanin.py``` which implement DDPG. Important training steps are in ```/gym/envs/mujoco/chase.py``` 

To cite this platform: 
```
@misc{Ant_racer,
  howpublished = {\href{https://github.com/niart/Ant_racer}{N. Wang, Ant_racer: a multi-agent pursuit-evasion platform. Github Repository, 2021, https://github.com/niart/Ant_racer}},} 
```  
An overview of Ant_racer game:
<p align="center">
<img src="https://github.com/niart/Ant_racer/blob/e65aa00da53000029a892883fec9e51d56977933/Screenshot%20from%202023-03-26%2001-01-37.png" width=60% height=60%>
</p>
