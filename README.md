# UnCAPTCHA
Using deep learning to beat text-in-image-based CAPTCHA! Trained and tested with data from [Kaggle](https://www.kaggle.com/datasets/fanbyprinciple/captcha-images).

## Demo

Try out our demo on HuggingFace! 

![Image of UnCaptcha Demo](images/UnCaptcha.png)

## Introduction
Text-based CAPTCHA, Completely Automated Public Turing tests to tell Computers and Humans Apart, are meant to be a defense mechanism against bots. We have created two models that can consistently beat text-based CAPTCHAs within two to three attempts. The first model is a segmentation-based convolutional neural network which uses edge dectection to segment CAPTCHAs into their individual characters, and classify each character indiviidually. The second model performs Optical Character Recognition (OCR) of full, multi-letter CAPTCHAs using a convolutional recurrent neural network and Connectionist Temporal Classification (CTC) loss. We hope to prove that text-based CAPTCHAs no longer serve their purpose and should be replaced with better alternatives such as ReCAPTCHA. 

![Image of UnCaptcha Poster](images/UnCaptcha.png)

## Results

## Instructions to Run

### 1. Preprocessing

### 2. Running the models

To train and test the segmentation model, simply run `code/main_segmentation.py`. This will save a trained model to `models/segmented`, and print out the reported accuracy. 
