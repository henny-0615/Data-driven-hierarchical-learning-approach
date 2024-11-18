You may find more details in our [Paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762401145X).

This repo contains a concrete implementation of the method in the paper.

# Data-driven-hierarchical-learning-approach
Data-driven hierarchical learning approach for multi-point servo control of Pan – Tilt – Zoom cameras
![image](https://github.com/henny-0615/Data-driven-hierarchical-learning-approach/blob/main/assets/main.png)

## Training
1. Install the necessary dependencies. Note that we conducted our experiments using python 3.8.10.
```html<div style="background-color: #f0f0f0; padding: 10px;">
pip install -r requirements\requirements.txt
```
2. Train the agent.
```html<div style="background-color: #f0f0f0; padding: 10px;">
python train.py
```
## Additional Useful Information  
Configuration [files](https://drive.google.com/drive/folders/118OspgB4jOaqOzKRI6am5WIuFLVxJD3V?usp=sharing) for controlling the PTZ camera and 
a simulation [environment](https://drive.google.com/drive/folders/1f3L6oyLOSDXIoXSdqLRRrdrR7PhgbGD7?usp=sharing) developed based on Unity are available.

## Acknowledgement
Thanks the authors for their works:  
[CORL](https://github.com/tinkoff-ai/CORL)  
[MatchFormer](https://github.com/jamycheung/MatchFormer)
