# Visual_tracking-based-on-CNN-and-RNN
MOTIVATION
The key challenge in building a neural net is requirement of humongous training data. The next hindrance to deploy the model in embedded AI device is number of computaions,
thus demanding a lighter model. We hypothesized that training Re3 visual tracker with less number of videos should also provide an appreciable level of tracking. Our 
hypothesis is backed up by the core idea of Re3, structured to learn tracking through feature level differences between consecutive frames of training videos.

ABOUT THE MODEL:
The training dataset for our project is 15 videos from imagenet video dataset. The model is implemented in Pytorch framework against the author`s implementation in tensorflow.
Since Pytorch is easy for debugging and maintenance, this framework was chosen. 

RESULT:
Model`s behaviour of tracking animals can be seen below. However, the model could not track a hand thus failing to generalize. Our training videos consist of animals like horse, panda,
etc.., as object of interest. It could be the reason behind model`s behaviour on animal videos. Also, the CNN model was frozen during training.

FUTURE SCOPE:
1) Unfreeze the CNN model and see its impact on generalization.
2) Analyze the behaviour of the model with lighter CNN nets like resnet.