Interpretability of CNN with ImageNet class index



Introduction

This project aims to investigate the interpretability of a Convolutional Neural Network with the help of pretrained CNN model; ResNet18, a Class Atribution Map; torch-cam, and a dataset containing 1000 classes; ImageNet.



Method

For this examination seven images where used.

The first fase of the investigation was to produce a positive and a negative image for three classes. The chosen classes were "African elephant", "Fire engine" and "Acoustic guitar". Three positive images[1][2][3] were porduced. For the Negative images, the correct classification would be "Hippopotamus", "Ambulance" and "Violin".[4][5][6]
These Negatives were chosen for there similar traits to the positives. Aiming to produce the most possible confusion for the model. 
These six images were run though the CAM_extractor with target_layer "layer4" and visualized using matplotlibs imshow().

The produced cams where overlayed ove the original images to give a cleared visualization of the results.

A Predict_class function was created which used the ImageNet Class Index to return a dictionary containing the predicted class index, class id, class name and confidence of prediction.
Each image was run through the function and the resulting predictions printed.

The second fase of the investigation looked at the different layers of the cam. Using two of the positive images, all four target layers' cams were produced and visualized side by side to show the progression of the layers. This fase aimed to investigate the different details the CNN would focus on for the classification.

The third fase of the investigation used an image which class was not included in the ImageNet Class Index [7]. The image was run through the layer4 cam_extractor and predict_class function. This investigation aimed to deduce the reasoning for the CNN's classification of an unknown class.



Results

The ResNet18 model predicted The Images as follows:

![Elephant_cam](Images/elephant_cam.png)

Image1: {'class_index': '386', 'class_id': 'n02504458', 'class_name': 'African_elephant', 'confidence': 0.5068271160125732}

![Hippo_cam](Images/hippo_cam.png)

Image2: {'class_index': '344', 'class_id': 'n02398521', 'class_name': 'hippopotamus', 'confidence': 0.999891996383667}

![Firetruck_cam](Images/firetruck_cam.png)

Image3: {'class_index': '555', 'class_id': 'n03345487', 'class_name': 'fire_engine', 'confidence': 0.9979885816574097}

![Ambulance_cam](Images/ambulance_cam.png)

Image4: {'class_index': '407', 'class_id': 'n02701002', 'class_name': 'ambulance', 'confidence': 0.9393348693847656}

![Guitar_cam](Images/guitar_cam.png)

Image5: {'class_index': '402', 'class_id': 'n02676566', 'class_name': 'acoustic_guitar', 'confidence': 0.9951629638671875}

![Violin_cam](Images/violin_cam.png)

Image6: {'class_index': '889', 'class_id': 'n04536866', 'class_name': 'violin', 'confidence': 0.9552302956581116}

![Statue_cam](Images/statue_cam.png)

Image7: {'class_index': '708', 'class_id': 'n03903868', 'class_name': 'pedestal', 'confidence': 0.4729427993297577}


The side by side visual of the different layers:
![All_layers_CAM](Images/cams_all_layers_output.png)



Discussion

The model held a high confidence level, above 95% for the initial 6 images apart from the African elephant which was only predicted with a 50,68% confidence. This deviance is presumed to be due to the similarity between the African elephant and the Indian elephant in the class index.
The model correctly predicted all the images despite the similar traits in the positives and negatives.
The heatmaps show that for the elephant the identifying traits are the tusks and the trunk.
For the hippo it's the little ears.
For the fire engine it seems to be the side of the truck with a focus on the doors.
for the ambulance the focus is on the back doors and on the text on the side of the vehicle.
For the guitarr the focus is on the wooden body.
And for the violin the identifier seems to be the tailpiece.

Looking at the CAMs of the different layers for the African elephant and the acoustic guitarr we can derrive the focus of the different layers.
The first layer focused on features such as edges and textures. It is rather pixelated. But there is some semblance of the shape of the image objective.
The second layer gave more attention to the patterns and specified parts of the picture.
The third layer starts being less pixelated and more focused on the different parts of the objective.
The fourth layer is even less pixelated and has clearly located a classifying part of the objective.

The final image, predicted to be a pedestal, overlayed with the heatmap shows that the CNN focused on the leg of the statue. Possibly taking it's bent shape and the stone texture as similar traits to a pedestal. But the heatmap is scattered over the whole picture. Along with the poor confidence level it indicates that the model had a hard time identifying this image. But the prediction is not too bad. The is a similarity between the statue and a pedestal. Espesially in the stone texture.



Sources

[1] African elephant (https://unsplash.com/photos/brown-elephant-on-green-grass-field-during-daytime-P7L5011nD5s)

[2] Fire engine (https://unsplash.com/photos/a-fire-truck-on-the-street-egergv8fSG8)

[3] Acoustic guitar (https://unsplash.com/photos/brown-acoustic-guitar-C8gib_msapY)

[4] Hippopotamus (https://unsplash.com/photos/a-couple-of-hippos-standing-next-to-each-other-Q_HLuU0Xv8Q)

[5] Ambulance (https://unsplash.com/photos/an-ambulance-driving-down-a-city-street-next-to-tall-buildings-l_ov5A67rI4)

[6] Violin (https://unsplash.com/photos/brown-violin-on-white-textile-d9_2kPJBG0U)

[7] Statue (https://unsplash.com/photos/a-statue-of-a-person-ZR8WQ5fDbo8)