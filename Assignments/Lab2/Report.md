# Interpretability of CNN with ImageNet class index



## Introduction

This project investigates the interpretability of a Convolutional Neural Network using a pretrained CNN model, ResNet18, class activation mapping methods via torchcam and the ImageNet dataset containing 1000 object classes.



## Method

For this examination seven images where used.

The first phase of the investigation was to produce a positive and a negative image for three classes. The chosen classes were "Norwegian elkhound", "Pomegranate" and "Speedboat". Three positive images[1][2][3] were produced. For the Negative images not present in the classification index, the correct classification would be "Chinchilla", "Pithaya" and "Yacht".[4][5][6]
These Negatives were chosen for their similar traits to the positives. Intending for the model to produce the same classification for the positive and the negative. 
These six images were run through the CAM_extractor with target_layer "layer4" and visualized using matplotlibs imshow().

The produced cams where overlayed over the original images to give a clearer visualization of the results.

A Predict_class function was created which used the ImageNet Class Index to return a dictionary containing the predicted class index, class id, class name and confidence of prediction.
Each image was run through the function and the resulting predictions printed.

The second phase of the investigation looked at the different layers of the cam. Using two of the positive images, all four target layers' attribution maps were produced and visualized side by side to show the progression of the layers. This phase aimed to investigate the different details the CNN would focus on for the classification.

The third phase of the investigation used an image which object was an imaginary figure[7]. The image would not be present in the index and there would not be any related species for the model to predict. The image was run through the layer4 cam_extractor and predict_class function. This investigation aimed to analyze how the model responds to unknown class inputs and to infer the reasoning behind its classification decision.



## Results

The ResNet18 model predicted The Images as follows:

![Elkhound_cam](Images/elkhound_cam.png)

Image1: {'class_index': 174, 'class_id': 'n02091467', 'class_name': 'Norwegian_elkhound', 'confidence': 0.8044467568397522}

![Chinchilla_cam](Images/chinchilla_cam.png)

Image2: {'class_index': 174, 'class_id': 'n02091467', 'class_name': 'Norwegian_elkhound', 'confidence': 0.7822440266609192}

![Pomegranate_cam](Images/pomegranate_cam.png)

Image3: {'class_index': 957, 'class_id': 'n07768694', 'class_name': 'pomegranate', 'confidence': 0.9994410872459412}

![Pithaya_cam](Images/pithaya_cam.png)

Image4: {'class_index': 957, 'class_id': 'n07768694', 'class_name': 'pomegranate', 'confidence': 0.133589506149292}

![Speedboat_cam](Images/speedboat_cam.png)

Image5: {'class_index': 814, 'class_id': 'n04273569', 'class_name': 'speedboat', 'confidence': 0.9994126558303833}

![Yacht_cam](Images/yacht_cam.png)

Image6: {'class_index': 814, 'class_id': 'n04273569', 'class_name': 'speedboat', 'confidence': 0.6126159429550171}

![Dragon_cam](Images/dragon_cam.png)

Image7: {'class_index': 354, 'class_id': 'n02437312', 'class_name': 'Arabian_camel', 'confidence': 0.867834746837616}

The side by side visual of the different layers:

![All_layers_CAM](Images/all_layers_cam.png)



## Discussion

The pomegranate image and the speedboat image were predicted with a 99.9% confidence. An expected result for a positive image. The Norwegian elkhound however, despite being a positive image only predicted with a 80.4% confidence. This could be acounted to the similarities between several different dog species present in the index. The Attribution map of the image confirms that the model was not particularily confused. The map is consentrated on the head of the dog with no scattered focus.
The negative image of a chinchilla held an interesting confidence of 78%. The model was almost as cirtain that the chinchilla was a norwegian elkhound as it was the actual norwegian elkhound. Inspecting the heatmap for the chinchilla it appears the nose and the tail are the main focus. One could argue that there is a resemblence between the tails of the two animals. However, there is not much similarity in the facial features or ears. Thus the confidence level of almost 80% seems gratuitus.
The negatives had more interesting results overall. The Pithaya's prediction of class pomegranate had a confidence of 13.4%. The attribution maps of the pomegranate and the pithaya shows that the model focused on the skins of the fruits. The insides and seeds were not particularily interesting, and the protruding elements were only mildly interesting. It seems that the color of the skin was the main attribute which led to the prediction for both of the images.
The yacht, sharing a lot of visual similarities with the speedboat was, not unexpectedly, predicted with a 61.3% confidence. The attribution maps show that where for the speedboat the focus lies mainly on the engine and the rear ead of the boat, for the yacht the focus is spread all over the object, indicating some confusion on the models behalf.
The prediction of the yacht, though incorrect, is at least on the right track with a vessel on the water such as the speedboat. The predictions for the pithaya and especially the chinchilla are completely incorrect and can not be of any use.


Looking at the CAMs of the different layers for the Norwegian elkhound and the pomegranate we can derrive the focus of the different layers.
The first layer focused on features such as edges and textures. It is rather pixelated. But there is some semblance of the shape and outline of the image objective.
The second layer gave more attention to the patterns and specified parts of the picture.
The third layer starts being less pixelated and more focused on the different parts of the objective.
The fourth layer is even less pixelated and has clearly located a classifying part of the objective.

The final image, not part of the class index, incorrectly predicted to be an Arabian camel, overlayed with the heatmap shows that the models primary focus was the wing of the dragon. Possibly due to its curved shape, the model could presume a resemblance with features commonly associated with the camel. But the heatmap is scattered over the whole picture, indicating some confusion for the model. The high confidence level of 86.8% is not the expected result. There is not much visual similarity between the image and its predicted class. However, since the image shows a statue of an imaginary animal, the fact that the model predicted another animal and not something with similar stone-like texture is to the models credit.



## Sources

[1] Norwegian elkhound (https://unsplash.com/photos/a-dog-laying-in-the-grass-on-a-sunny-day-fSb9bj7-25k)

[2] Pomegranate (https://unsplash.com/photos/red-round-fruit-on-black-surface-SCMnIJV3DrQ)

[3] Speedboat (https://unsplash.com/photos/white-and-red-boat-on-water-during-daytime-y0Br6D28Lkg)

[4] Chinchilla (https://unsplash.com/photos/person-holding-gray-and-white-rabbit-6vU-kvHdGRI)

[5] Pithaya (https://unsplash.com/photos/a-dragon-fruit-cut-in-half-on-a-white-background-IbRkYrIgrWo)

[6] Yacht (https://unsplash.com/photos/birds-eye-photography-of-yacht-on-body-of-water-XZOO6QHub60)

[7] Dragon (https://unsplash.com/photos/a-dragon-statue-on-top-of-a-building-eFyVtxyQA34)