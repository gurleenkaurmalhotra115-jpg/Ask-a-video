# Label Verification Report

**Source File:** `eval/questions.csv`
**Total Answerable Questions:** 10

---

### Row #1: How many neurons are in the input layer for the handwritten digit example and why?
- **Video ID:** `aircAruvnKk` | **Type:** `factual`
- **Main Span:** `03:08` - `04:02` (54.0s)
- **Transcript Text:**
  > For example the network starts with a bunch of neurons corresponding to each of the 28x28 pixels of the input image, which is 784 neurons in total. Each one of these holds a number that represents the grayscale value of the corresponding pixel, ranging from 0 for black pixels up to 1 for white pixels. This number inside the neuron is called its activation, and the image you might have in mind here is that each neuron is lit up when its activation is a high number. So all of these 784 neurons make up the first layer of our network. Now jumping over to the last layer, this has 10 neurons, each representing one of the digits. The activation in these neurons, again some number that's between 0 and 1, represents how much the system thinks that a given image corresponds with a given digit.
- **Alt Span:** `04:58` - `05:40` (42.0s)
- **Alt Transcript Text:**
  > Now the network I'm showing here has already been trained to recognize digits, and let me show you what I mean by that. It means if you feed in an image, lighting up all 784 neurons of the input layer according to the brightness of each pixel in the image, that pattern of activations causes some very specific pattern in the next layer which causes some pattern in the one after it, which finally gives some pattern in the output layer. And the brightest neuron of that output layer is the network's choice, so to speak, for what digit this image represents. And before jumping into the math for how one layer influences the next, or how training works, let's just talk about why it's even reasonable

### Row #2: What does an activation number inside a neuron represent?
- **Video ID:** `aircAruvnKk` | **Type:** `factual`
- **Main Span:** `03:15` - `04:00` (45.0s)
- **Transcript Text:**
  > Each one of these holds a number that represents the grayscale value of the corresponding pixel, ranging from 0 for black pixels up to 1 for white pixels. This number inside the neuron is called its activation, and the image you might have in mind here is that each neuron is lit up when its activation is a high number. So all of these 784 neurons make up the first layer of our network. Now jumping over to the last layer, this has 10 neurons, each representing one of the digits. The activation in these neurons, again some number that's between 0 and 1, represents how much the system thinks that a given image corresponds with a given digit.

### Row #3: What do the neurons in the output layer represent?
- **Video ID:** `aircAruvnKk` | **Type:** `factual`
- **Main Span:** `03:45` - `04:20` (35.0s)
- **Transcript Text:**
  > Now jumping over to the last layer, this has 10 neurons, each representing one of the digits. The activation in these neurons, again some number that's between 0 and 1, represents how much the system thinks that a given image corresponds with a given digit. There's also a couple layers in between called the hidden layers, which for the time being should just be a giant question mark for how on earth this process of recognizing digits is going to be handled. In this network I chose two hidden layers, each one with 16 neurons, and admittedly that's kind of an arbitrary choice.

### Row #4: Why does each connection between neurons have an assigned weight?
- **Video ID:** `aircAruvnKk` | **Type:** `conceptual`
- **Main Span:** `09:05` - `10:00` (55.0s)
- **Transcript Text:**
  > or the pattern that several edges can make a loop, and other such things? Well, what we'll do is assign a weight to each one of the connections between our neuron and the neurons from the first layer. These weights are just numbers. Then take all of those activations from the first layer and compute their weighted sum according to these weights. I find it helpful to think of these weights as being organized into a little grid of their own, and I'm going to use green pixels to indicate positive weights, and red pixels to indicate negative weights, where the brightness of that pixel is some loose depiction of the weight's value. Now if we made the weights associated with almost all of the pixels zero except for some positive weights in this region that we care about, then taking the weighted sum of all the pixel values really just amounts to adding up the values of the pixel just in the region that we care about. And if you really wanted to pick up on whether there's an edge here,

### Row #5: What is the role of a bias in a neural network neuron?
- **Video ID:** `aircAruvnKk` | **Type:** `conceptual`
- **Main Span:** `11:00` - `11:50` (50.0s)
- **Transcript Text:**
  > light up when the weighted sum is bigger than 0. Maybe you only want it to be active when the sum is bigger than say 10. That is, you want some bias for it to be inactive. What we'll do then is just add in some other number like negative 10 to this weighted sum before plugging it through the sigmoid squishification function. That additional number is called the bias. So the weights tell you what pixel pattern this neuron in the second layer is picking up on, and the bias tells you how high the weighted sum needs to be before the neuron starts getting meaningfully active. And that is just one neuron. Every other neuron in this layer is going to be connected to all 784 pixel neurons from the first layer, and each one of those 784 connections has its own weight associated with it.

### Row #6: What does the sigmoid squishification function do?
- **Video ID:** `aircAruvnKk` | **Type:** `conceptual`
- **Main Span:** `10:25` - `11:10` (45.0s)
- **Transcript Text:**
  > So a common thing to do is to pump this weighted sum into some function that squishes the real number line into the range between 0 and 1. And a common function that does this is called the sigmoid function, also known as a logistic curve. Basically very negative inputs end up close to 0, positive inputs end up close to 1, and it just steadily increases around the input 0. So the activation of the neuron here is basically a measure of how positive the relevant weighted sum is. But maybe it's not that you want the neuron to light up when the weighted sum is bigger than 0. Maybe you only want it to be active when the sum is bigger than say 10. That is, you want some bias for it to be inactive.

### Row #7: Why is ReLU preferred over the sigmoid function in modern networks?
- **Video ID:** `aircAruvnKk` | **Type:** `hard`
- **Main Span:** `17:30` - `18:25` (55.0s)
- **Transcript Text:**
  > Exactly. But relatively few modern networks actually use sigmoid anymore. Yeah. It's kind of old school right? Yeah or rather ReLU seems to be much easier to train. And ReLU, ReLU stands for rectified linear unit? Yes it's this kind of function where you're just taking a max of zero and a where a is given by what you were explaining in the video and what this was sort of motivated from I think was a partially by a biological analogy with how neurons would either be activated or not. And so if it passes a certain threshold it would be the identity function but if it did not then it would just not be activated so it'd be zero so it's kind of a simplification. Using sigmoids didn't help training or it was very difficult to train at some point and people just tried ReLU and it happened to work very well for these incredibly deep neural networks.

### Row #8: How many total weights and biases or parameters are in this network?
- **Video ID:** `aircAruvnKk` | **Type:** `factual`
- **Main Span:** `12:15` - `12:45` (30.0s)
- **Transcript Text:**
  > a bunch of weights and biases associated with them. All said and done, this network has almost exactly 13,000 total weights and biases. 13,000 knobs and dials that can be tweaked and turned to make this network behave in different ways. So when we talk about learning, what that's referring to is getting the computer to find a valid setting for all of these many many numbers so that it'll actually solve the problem at hand. One thought experiment that is at once fun and kind of horrifying is to imagine sitting
- **Alt Span:** `15:45` - `16:10` (25.0s)
- **Alt Transcript Text:**
  > 784 numbers as an input and spits out 10 numbers as an output. It's an absurdly complicated function, one that involves 13,000 parameters in the forms of these weights and biases that pick up on certain patterns, and which involves iterating many matrix vector products and the sigmoid squishification function, but it's just a function nonetheless. And in a way it's kind of reassuring that it looks complicated. I mean if it were any simpler, what hope would we have that it could take on the challenge of recognizing digits?

### Row #9: How does the network combine weights activations and bias mathematically?
- **Video ID:** `aircAruvnKk` | **Type:** `hard`
- **Main Span:** `11:40` - `12:15` (35.0s)
- **Transcript Text:**
  > Every other neuron in this layer is going to be connected to all 784 pixel neurons from the first layer, and each one of those 784 connections has its own weight associated with it. Also, each one has some bias, some other number that you add on to the weighted sum before squishing it with the sigmoid. And that's a lot to think about! With this hidden layer of 16 neurons, that's a total of 784 times 16 weights, along with 16 biases. And all of that is just the connections from the first layer to the second. The connections between the other layers also have a bunch of weights and biases associated with them.

### Row #10: Why is the neural network structured in layers to break down images?
- **Video ID:** `aircAruvnKk` | **Type:** `conceptual`
- **Main Span:** `06:00` - `07:00` (60.0s)
- **Transcript Text:**
  > An 8 also has a loop up top, but it's paired with another loop down low. A 4 basically breaks down into three specific lines, and things like that. Now in a perfect world, we might hope that each neuron in the second to last layer corresponds with one of these subcomponents, that anytime you feed in an image with, say, a loop up top, like a 9 or an 8, there's some specific neuron whose activation is going to be close to 1. And I don't mean this specific loop of pixels, the hope would be that any generally loopy pattern towards the top sets off this neuron. That way, going from the third layer to the last one just requires learning which combination of subcomponents corresponds to which digits. Of course, that just kicks the problem down the road, because how would you recognize these subcomponents, or even learn what the right subcomponents should be? And I still haven't even talked about how one layer influences the next, but run with me on this one for a moment. Recognizing a loop can also break down into subproblems. One reasonable way to do this would be to first recognize the various little edges that make it up.

---
## Summary

- **Total Rows:** 10
- **Total Warnings:** 0
