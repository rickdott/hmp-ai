# Classifying cognitive processing stages
Following research on cognitive processing stages and localizing their onset:

J. R. Anderson, Q. Zhang, J. P. Borst, and M. M. Walsh, “The discovery of processing stages: Extension of Sternberg’s method.,” Psychological Review, vol. 123, no. 5, pp. 481–509, Oct. 2016, doi: 10.1037/rev0000030.

J. P. Borst and J. R. Anderson, “The discovery of processing stages: Analyzing EEG data with hidden semi-Markov models,” NeuroImage, vol. 108, pp. 60–73, Mar. 2015, doi: 10.1016/j.neuroimage.2014.12.029.

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

Based on [HMP](https://github.com/GWeindel/hmp) created by Gabriel Weindel\
Written by Rick den Otter\
r.denotter@students.uu.nl\
rickdotyt@gmail.com