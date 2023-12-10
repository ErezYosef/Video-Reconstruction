  
## Video Reconstruction from a Single Motion Blurred Image using Learned Dynamic Phase Coding  
### [Erez Yosef](https://erezyosef.github.io/), [Shay Elmalem](https://scholar.google.com/citations?user=4N4ToIkAAAAJ&hl=en) & [Raja Giryes](https://www.giryes.sites.tau.ac.il/)

##### This repo contains a demo code for our paper _Video Reconstruction from a Single Motion Blurred Image using Learned Dynamic Phase Coding_.

##### All images in `input` folder are real captures using our designed camera, and this demo code reconstruct a video sequence of the captured scene from a single image.

#### [Paper](https://www.nature.com/articles/s41598-023-40297-0)  

<a href="http://www.youtube.com/watch?feature=player_embedded&v=rYS9pnGXjBU
" target="_blank"><img src="inputs/car.png" 
alt="Watch the video" width="240" /></a>

## Abstract    
Video reconstruction from a single motion-blurred image is a challenging problem, which can enhance the capabilities of existing cameras. Recently, several works addressed this task using conventional imaging and deep learning. Yet, such purely digital methods are inherently limited, due to direction ambiguity and noise sensitivity. Some works attempt to address these limitations with non-conventional image sensors, however, such sensors are extremely rare and expensive. To circumvent these limitations by simpler means, we propose a hybrid optical-digital method for video reconstruction that requires only simple modifications to existing optical systems. We use learned dynamic phase-coding in the lens aperture during image acquisition to encode motion trajectories, which serve as prior information for the video reconstruction process. The proposed computational camera generates a sharp frame burst of the scene at various frame rates from a single coded motion-blurred image, using an image-to-video convolutional neural network. We present advantages and improved performance compared to existing methods, with both simulations and a real-world camera prototype. We extend our optical coding to video frame interpolation and present robust and improved results for noisy videos.  


## Installation  
-------------  
Using Python 3.7 is preffered.
Please install the dependencies by runinng:
```  
pip install -r requirements.txt  
```  
  
## Video reconstruction  
  
To generate video frames, run the command:  
```
python run_demo.py --lf [weights_file_path] -t [num_samples(frames)] --imgpath [path_to_img] --batch_size [BS]
```
where: 

- lf = (load file) path to weights .pt file (defaoult: final_weights.pt )  
- t = how many frames to generate from a single image. default 25  
- batch_size = default 2  
- imgpath = path to the input img
  
The generated frames will be generated in `outputs/[img_name]` folder.
### For example, you can run:  
```python run_demo.p -t 25 --imgpath inputs/car2.png```

### If you find our work useful in your research or publication, please cite it:


```
@article{yosef2023video,
  title={Video reconstruction from a single motion blurred image using learned dynamic phase coding},
  author={Yosef, Erez and Elmalem, Shay and Giryes, Raja},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={13625},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

#### [Erez Yosef](https://erezyosef.github.io/)
------