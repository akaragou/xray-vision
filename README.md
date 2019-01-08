# X-Ray Vision
X-Ray vision is a web app where a radiologist can upload an X-Ray and receive the probability that an X-Ray contains an abnormality along with the highlighted region of abnormality 

![plot](https://user-images.githubusercontent.com/16754088/46711409-3d08c100-cc1a-11e8-9187-d2a9097add35.png)
---
- models - folder that contains all models that were trianed: DenseNets, SEResNets and ResNets
- notebook - folder that contains ipython notebook with exploratory data analysis
- process_results - folder where input image and generate heatmap are placed
- static - folder that contains files and scripts used for web app such as images, javascript and css
- templates - folder that contains html for hompage and processed results page 
- uploads - folder that contains user uploaded X-Rays
- app.py - fask web app where a radiologist can upload an image and receive the probability that the image contains an abnormality along with the highlighted region of abnormality 
- bone\_type\_class\_weights.npy - weights for the different class of bone types: elbow, wrist, shoulder etc.
- cleanup.sh - script move all summaries, results and checkpoints to S3
- config.py - script that configures hyperparameters and augmentations
- data\_utils.py - script for data augmentations and encoding and decoding tfrecords
- freeze\_graph.py - script to freeze a graph and store it as a protobuf
- frozen\_graph.pb - frozen DenseNet 121 protobuf
- prepare\_tf\_records.py - script to encode image data into tfrecord format
- target\_class\_weights.npy - weights for abnormal and normal class
- test.py - script to test trained model performance
- train.py - script to train model 

---
## App Functionality 

A user can upload an image:

![plot](https://user-images.githubusercontent.com/16754088/46711410-3da15780-cc1a-11e8-86a3-3bd00710cd4e.png)

Then the user will be taken to a new page were the input X-Ray, heatmap of abnormality, and proability of abnormality will be provided. Several examples are given bellow:

![plot](https://user-images.githubusercontent.com/16754088/46711411-3da15780-cc1a-11e8-8aca-ac13e413a2af.png)
![plot](https://user-images.githubusercontent.com/16754088/46711412-3da15780-cc1a-11e8-8b4f-4c514ac7d8bf.png)
![plot](https://user-images.githubusercontent.com/16754088/46711413-3da15780-cc1a-11e8-8f69-e57b71488f3e.png)

---

## Results From Trained Models

![plot](https://user-images.githubusercontent.com/16754088/50789160-725c9380-1289-11e9-93fa-5ad5fcb4d3ed.png)