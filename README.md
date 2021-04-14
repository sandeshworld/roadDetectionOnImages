# roadDetectionOnImages
Deep CNN on Images for Road Detection using Jupyter Notebook

Our model is contained in the ![testing_new.ipynb](testing_new.ipynb) file.

Checkout our final paper on using the U-Net technique for road detection:

![FINAL PAPER](drivableAreaFinalReport.pdf)

Our model takes in images from the Berkeley Self Driving Dataset, compresses it, and then identifies the pixels which are part of the drivable area on the image.

![image](https://user-images.githubusercontent.com/28467603/114770219-0d256180-9d20-11eb-8f58-da3657e7a2be.png)

This image effectively shows the output of our model.

The top image is the actual label (Drivable area).
The second image is result of our model after we input the dashcam RGB image. 
The third image is the actual dashcam RGB image.
The final image is the output of our model after we apply threshold filtering. 

As you can see, our model is generally effective at predicting the "drivable pixels" on the image.
