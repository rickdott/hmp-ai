# Code repository for thesis on classifying processing stages
Plan:
- Find data from EEG experiments where bump localizing is already done
	- Divide data into 'chunks' based on bump location, labeling according to experiment
		- First is perceptual stage, second is x stage, third is y stage, fourth is motor stage
- Train different models (CNN, RNN(LSTM), DBN) to see if they classify above chance
	- First train on perception experiment 1
		- One participant first? Try to prevent overfitting
	- Then train on two similar experiments together, see if it generalizes across both experiments testing the same thing done at different labs
	- Train on one, test on other?
- Then train on more classes (include experiment 3, AR/LDM)

Written by Rick den Otter\
r.denotter@students.uu.nl\
rickdotyt@gmail.com