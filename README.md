# AWS ML deepComposer:
AWS Deep Composer uses Generative AI, or specifically Generative Adversarial Networks (GANs), to generate music. GANs pit 2 networks, a generator and a discriminator, against each other to generate new content .The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. In this context, the generator is like the orchestra and the discriminator is like the conductor. The orchestra plays and generates the music. The conductor judges the music created by the orchestra and coaches the orchestra to improve for future iterations. So an orchestra, trains, practices, and tries to generate music, and then the conductor coaches them to produced more polished music.

Orchestra and Conductor as a metaphor for GANS

# GANS is Similar to an Orchestra and a Conductor: 
The More They Work Together, the Better They Can Perform!
AWS DeepComposer Workflow Use the AWS DeepComposer keyboard or play the virtual keyboard in the AWS DeepComposer console to input a melody.Use a model in the AWS DeepComposer console to generate an original musical composition. You can choose from jazz, rock, pop, symphony or Jonathan Coulton pre-trained models or you can also build your own custom genre model in Amazon SageMaker.Publish your tracks to SoundCloud or export MIDI files to your favorite Digital Audio Workstation (like Garage Band) and get even more creative.

# Compose Music with AWS DeepComposer Models
Now that you know a little more about AWS DeepComposer including its workflow and what GANs are, let’s compose some music with AWS DeepComposer models. We’ll begin this demonstration by listening to a sample input and a sample output, then we’ll explore DeepComposer’s music studio, and we’ll end by generating a composition with a 4 part accompaniment.
- To get to the main AWS DeepComposer console, navigate to AWS DeepComposer. Make sure you are in the US East-1 region.
- Once there, click on Get started
- In the left hand menu, select Music studio to navigate to the DeepComposer music studio
- To generate music you can use a virtual keyboard or the physical AWS DeepComposer keyboard. For this lab, we’ll use the virtual keyboard.
- To view sample melody options, select the drop down arrow next to Input
- Select Twinkle, Twinkle, Little Star
- Next, choose a model to apply to the melody by clicking Select model
- From the sample models, choose Rock and then click Select model
- Next, select Generate composition. The model will take the 1 track melody and create a multitrack composition (in this case, it created 4 tracks)
- Click play to hear the output
# Generative AI
Generative AI has been described as one of the most promising advances in AI in the past decade by the MIT Technology Review.

Generative AI opens the door to an entire world of creative possibilities with practical applications emerging across industries, from turning sketches into images for accelerated product development, to improving computer-aided design of complex objects.

For example, Glidewell Dental is training a generative adversarial network adept at constructing detailed 3D models from images. One network generates images and the second inspects those images. This results in an image that has even more anatomical detail than the original teeth they are replacing.

Glidewell Dental is training GPU powered GANs to create dental crown models
Glidewell Dental is training GPU powered GANs to create dental crown models

Generative AI enables computers to learn the underlying pattern associated with a provided input (image, music, or text), and then they can use that input to generate new content. Examples of Generative AI techniques include Generative Adversarial Networks (GANs), Variational Autoencoders, and Transformers.

# What are GANs?
GANs, a generative AI technique, pit 2 networks against each other to generate new content. The algorithm consists of two competing networks: a generator and a discriminator.

A generator is a convolutional neural network (CNN) that learns to create new data resembling the source data it was trained on.

The discriminator is another convolutional neural network (CNN) that is trained to differentiate between real and synthetic data.

The generator and the discriminator are trained in alternating cycles such that the generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

# Like the collaboration between an orchestra and its conductor
The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. An orchestra doesn’t create amazing music the first time they get together. They have a conductor who both judges their output, and coaches them to improve. So an orchestra, trains, practices, and tries to generate polished music, and then the conductor works with them, as both judge and coach.

The conductor is both judging the quality of the output (were the right notes played with the right tempo) and at the same time providing feedback and coaching to the orchestra (“strings, more volume! Horns, softer in this part! Everyone, with feeling!”). Specifically to achieve a style that the conductor knows about. So, the more they work together the better the orchestra can perform.The Generative AI that AWS DeepComposer teaches developers about uses a similar concept. We have two machine learning models that work together in order to learn how to generate musical compositions in distinctive styles.

# Training a machine learning model using a dataset of Bach compositions
AWS DeepComposer uses GANs to create realistic accompaniment tracks. When you provide an input melody, such as twinkle-twinkle little star, using the keyboard U-Net will add three additional piano accompaniment tracks to create a new musical composition.The U-Net architecture uses a publicly available dataset of Bach’s compositions for training the GAN. In AWS DeepComposer, the generator network learns to produce realistic Bach-syle music while the discriminator uses real Bach music to differentiate between real music compositions and newly created ones

# generator
The generator network used in AWS DeepComposer is adapted from the U-Net architecture, a popular convolutional neural network that is used extensively in the computer vision domain. The network consists of an “encoder” that maps the single track music data (represented as piano roll images) to a relatively lower dimensional “latent space“ and a ”decoder“ that maps the latent space back to multi-track music data.

### Here are the inputs provided to the generator:

- Single-track piano roll: A single melody track is provided as the input to the generator.
- Noise vector: A latent noise vector is also passed in as an input and this is responsible for ensuring that there is a flavor to each output generated by the generator, even when the same input is provided.
As described in previous sections, a GAN consists of 2 networks: a generator and a discriminator. Let’s discuss the generator and discriminator networks used in AWS DeepComposer.

Generator
The generator network used in AWS DeepComposer is adapted from the U-Net architecture, a popular convolutional neural network that is used extensively in the computer vision domain. The network consists of an “encoder” that maps the single track music data (represented as piano roll images) to a relatively lower dimensional “latent space“ and a ”decoder“ that maps the latent space back to multi-track music data.

Here are the inputs provided to the generator:

Single-track piano roll: A single melody track is provided as the input to the generator.
Noise vector: A latent noise vector is also passed in as an input and this is responsible for ensuring that there is a flavor to each output generated by the generator, even when the same input is provided.
U-Net Architecture diagram
Notice that the encoding layers of the generator on the left side and decoder layer on on the right side are connected to create a U-shape, thereby giving the name U-Net to this architecture

# Discriminator
The goal of the discriminator is to provide feedback to the generator about how realistic the generated piano rolls are, so that the generator can learn to produce more realistic data. The discriminator provides this feedback by outputting a scalar value that represents how “real” or “fake” a piano roll is.

Since the discriminator tries to classify data as “real” or “fake”, it is not very different from commonly used binary classifiers. We use a simple architecture for the critic, composed of four convolutional layers and a dense layer at the end.

# Evaluation
Typically when training any sort of model, it is a standard practice to monitor the value of the loss function throughout the duration of the training. The discriminator loss has been found to correlate well with sample quality. You should expect the discriminator loss to converge to zero and the generator loss to converge to some number which need not be zero. When the loss function plateaus, it is an indicator that the model is no longer learning. At this point, you can stop training the model. You can view these loss function graphs in the AWS DeepComposer console.

Sample output quality improves with more training
After 400 epochs of training, discriminator loss approaches near zero and the generator converges to a steady-state value. Loss is useful as an evaluation metric since the model will not improve as much or stop improving entirely when the loss plateaus.

While standard mechanisms exist for evaluating the accuracy of more traditional models like classification or regression, evaluating generative models is an active area of research. Within the domain of music generation, this hard problem is even less well-understood.

To address this, we take high-level measurements of our data and show how well our model produces music that aligns with those measurements. If our model produces music which is close to the mean value of these measurements for our training dataset, our music should match the general “shape”. You’ll see graphs of these measurements within the AWS DeepComposer console

### Here are a few such measurements:

Empty bar rate: The ratio of empty bars to total number of bars.
Number of pitches used: A metric that captures the distribution and position of pitches.
# build a custom GAN - part1 
In Scale Ratio: Ratio of the number of notes that are in the key of C, which is a common key found in music, to the total number of notes.
In this demonstration we’re going to synchronize what you’ve learned about software development practices and machine learning, using AWS DeepComposer to explore those best practices against a real life use case.

Coding Along With The Instructor (Optional)
To create the custom GAN, you will need to use an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the demo and build a custom GAN, you may incur a cost.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation

## Getting Started
- Setting Up the DeepComposer Notebook
- To get to the main Amazon SageMaker service screen, navigate to the AWS SageMaker console. You can also get there from within the AWS Management Console by searching for Amazon SageMaker.
- Once inside the SageMaker console, look to the left hand menu and select Notebook Instances.
- Next, click on Create notebook instance.
- In the Notebook instance setting section, give the notebook a name, for example, DeepComposerUdacity.
- Based on the kind of CPU, GPU and memory you need the next step is to select an instance type. For our purposes, we’ll configure a ml.c5.4xlarge
- Leave the Elastic Inference defaulted to none.
- In the Permissions and encryption section, create a new IAM role using all of the defaults.
- When you see that the role was created successfully, navigate down a little ways to the Git repositories section
- Select Clone a public Git repository to this notebook instance only
- Copy and paste the public URL into the Git repository URL section: https://github.com/aws-samples/aws-deepcomposer-samples
- Select Create notebook instance
- Give SageMaker a few minutes to provision the instance and clone the Git repository
- Exploring the Notebook
- Now that it’s configured and ready to use, let’s take a moment to investigate what’s inside the notebook

# Open the Notebook
Click Open Jupyter.
- When the notebook opens, click on Lab 2.
- When the lab opens click on GAN.ipynb.
- Review: Generative Adversarial Networks (GANs).
GANs consist of two networks constantly competing with each other:

- Generator network that tries to generate data based on the data it was trained on.
- Discriminator network that is trained to differentiate between real data and data which is created by the generator.
Note: The demo often refers to the discriminator as the critic. The two terms can be used interchangeably.

# Set Up the Project
- Run the first Dependencies cell to install the required packages
- Run the second Dependencies cell to import the dependencies
- Run the Configuration cell to define the configuration variables
- Note: While executing the cell that installs dependency packages, you may see warning messages indicating that later versions of conda are available for certain packages. It is completely OK to ignore this message. It should not affect the execution of this notebook.

### Good Coding Practices
- Do not hard-code configuration variables
- Move configuration variables to a separate config file
- Use code comments to allow for easy code collaboration

# Why Do We Need to Prepare Data?
Data often comes from many places (like a website, IoT sensors, a hard drive, or physical paper) and it’s usually not clean or in the same format. Before you can better understand your data, you need to make sure it’s in the right format to be analyzed. Thankfully, there are library packages that can help! One such library is called NumPy, which was imported into our notebook.

## Piano Roll Format
The data we are preparing today is music and it comes formatted in what’s called a “piano roll”. Think of a piano roll as a 2D image where the X-axis represents time and the Y-axis represents the pitch value. Using music as images allows us to leverage existing techniques within the computer vision domain.

Our data is stored as a NumPy Array, or grid of values. Our dataset comprises 229 samples of 4 tracks (all tracks are piano). Each sample is a 32 time-step snippet of a song, so our dataset has a shape of:

>> (num_samples, time_steps, pitch_range, tracks)
### Load and View the Dataset
- Run the next cell to play a song from the dataset.
- Run the next cell to load the dataset as a nympy array and output the shape of the data to confirm that it matches the (229, 32, 128, 4) shape we are expecting
- Run the next cell to see a graphical representation of the data.
# Create a Tensorflow Dataset
Much like there are different libraries to help with cleaning and formatting data, there are also different frameworks. Some frameworks are better suited for particular kinds of machine learning workloads and for this deep learning use case, we’re going to use a Tensorflow framework with a Keras library.

We'll use the dataset object to feed batches of data into our model.

- Run the first Load Data cell to set parameters.
- Run the second Load Data cell to prepare the data.
# Model Architecture
Before we can train our model, let’s take a closer look at model architecture including how GAN networks interact with the batches of data we feed it, and how they communicate with each other.
# How the Model Works
The model consists of two networks, a generator and a critic. These two networks work in a tight loop:

- The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
- The discriminator evaluates the generated music tracks and predicts how far they deviate from the real data in the training dataset.
- The feedback from the discriminator is used by the generator to help it produce more realistic music the next time.
- As the generator gets better at creating better music and fooling the discriminator, the discriminator needs to be retrained by using music tracks just generated by the generator as fake inputs and an equivalent number of songs from the original dataset as the real input.
- We alternate between training these two networks until the model converges and produces realistic music.
- The discriminator is a binary classifier which means that it classifies inputs into two groups, e.g. “real” or “fake” data.
balance with full details:https://classroom.udacity.com/courses/ud090/lessons/099925a2-4f01-41c7-a4d4-8ce246f7b801/concepts/45be01fd-ee02-4b5c-8826-b4e2e4082255
