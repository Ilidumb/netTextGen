Warning: it's all probably broken.
# netTextGen
Text Generation with TensorFlow using a Recurrent Neural Network

## Setup
The training_checkpoints folder is required, just drop it in and ignore it. If you want to use one of the pre-trained models, you have to move them out of the folder and into the base netTextGen folder.
## Custom training data
To use custom training data, move your .txt file into the trainData folder. IT HAS TO BE ENCODED IN UTF-8! You can also use links instead of file://, but keep in mind it has to be a raw website. It imports the html too.
## Customisation
Add more GRU / LSTM layers, change the amount of epochs. Afer training the model, its saved in .h5 format. Remeber to change the name to avoid file overrides. Don't use the model trained on [text1] on text[2], the character mappings won't be the same.
## netTextGenLoad
It loads your model and runs it, so you don't have to re-train it every single time. Remeber, netTextGenLoad requires a .txt file too!
