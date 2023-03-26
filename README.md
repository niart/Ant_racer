# Ant_racer, 
## a virtual Multi-agent pursuit-evasion platform based on OpenAI/Gym and Mujoco
This project is a variant of multi-agent game platform [TripleSumo](https://link.springer.com/chapter/10.1007/978-3-031-15908-4_15) (repository: https://github.com/niart/triplesumo). 

Steps of installing this platform:
1. Download Mujoco200 from: https://www.roboti.us/download.html, extract it in 
   ```/home/your_username/.mujoco/ ```
2. Add ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/your_username/.mujoco/mujoco200/bin``` to your ```~/.bashrc```, and then ```source ~/.bashrc```
3. git clone https://github.com/niart/Ant_racer.git and "cd Ant_racer"
4. Create a virtual environment 'Ant_racer' with ```conda env create -f ant_racer_env.yml''' 
5. Use the ```gym``` foler of this repository to replace the ```gym``` installed in your anaconda. 
6. To test the demo, run ```python chase_demo.py```. If you meet error ```Creating window glfw ... ERROR: GLEW initalization error: Missing GL version```, you may add ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so``` to ```~/.bashrc``` to solve it

A simple RL algorithm interface has been written in ```chase_runmanin.py``` which implement DDPG.
An overview of Ant_racer game:
<p align="center">
<img src="https://github.com/niart/Ant_racer/blob/e65aa00da53000029a892883fec9e51d56977933/Screenshot%20from%202023-03-26%2001-01-37.png" width=60% height=60%>
</p>
