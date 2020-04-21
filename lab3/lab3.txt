CS 179: GPU Computing
Assignment 3

Due: Wednesday, April 24, 2019 - 3:00 PM


Put all answers in a file called README.txt. After answering all of the
questions, list how long part 1 and part 2 took. Feel free to leave any other
feedback.


Submission: 
------------------

========================================
NOTE: New submission method!

Instead of emailing us the solution, put a zip file in your home directory 
on Titan, in the format:

lab3_2019_submission.zip


Your submission should be a single archive file (.zip) 
with your README file and all code.

========================================



Resource usage:
------------------

This is a classic problem. Please do not look up the solutions directly, except as described in the "reduction" resource below. (Feel free to look up general documentation.)





Question 1: Large-Kernel Convolution (coding) (50 pts)
--------------------------------------------------------
--------------------------------------------------------



Introduction:
------------------

On Friday, we saw that convolution using the Fast Fourier transform is far more computationally efficient when dealing with large impulse responses. The general pseudocode for the related "circular convolution" is given by (for vectors x, h):

	X = FFT(x)
	H = FFT(h)
	Y = (pointwise multiplication of X and H)
	y = IFFT(Y)


Even better, on Wednesday, we saw that the FFT is parallelizable, and is included as the CUDA library cuFFT.

(Recall that we must also zero-pad our data in order to do linear convolution, as well as to make X and H equal lengths. See Lecture 9 slides for details.)


Just like in Homework 1, where we accelerated the "small-kernel convolution", we now want to accelerate the FFT convolution, i.e. "large-kernel convolution", using this information.


To do:
------------------

Complete the GPU-accelerated FFT convolution by filling in the parts marked "TODO" in fft_convolve.cc and fft_convolve_cuda.cu.



Notes:
------------------

(For general notes, see "Assignment Notes" section at the bottom of the assignment.)

Many implementations of the DFT (cuFFT included) have the property that IDFT(DFT(x)) does not actually return the vector x, but instead Nx (where N is the length of x). Remember to scale by this factor in the product/scale kernel.

Remember to destroy the cuFFT plan when finished with transforms.




Question 2: Normalization (coding/explanation) (50 pts)
--------------------------------------------------------
--------------------------------------------------------


Introduction:
------------------

After convolution, the output signal can take on all sorts of values. However, WAV audio files, when using floating-point values, store their sequences of amplitudes in the range [-1, 1] (usually corresponding to the minimum and maximum voltages sent out from the sound card). Any values outside this range are "clipped" to the range, causing a highly unpleasant (and potentially dangerous) sound for the listener and for electronic equipment. 

To avoid this problem, we have to normalize our signal, typically done by dividing by the maximum amplitude's magnitude. We can parallelize this process on the GPU as a reduction problem, similar to that presented on Monday (Lecture 7) - this time, we aim to find the maximum (absolute value) of our data, instead of the sum. Once we find our maximum, we can divide to normalize our signal (also GPU-accelerable, similar to the very first array addition problem in Lectures 1 and 2).


To do:
------------------

Try to complete Question 1 first. (Some parts, such as memory transfers, are necessary for both convolution and normalization to function.)

Then, complete the GPU-accelerated normalization by filling in the parts marked "TODO 2" in fft_convolve.cc and fft_convolve_cuda.cu.

For the reduction problem (the maximum-finding kernel), please thoroughly explain your method of reduction and any optimizations you make (especially as they relate to GPU hardware). Leave this explanation in the comments, around where the reduction kernel is. 

There are many ways to do reduction (some of which we discussed in class), some of which perform better than others. Score will be determined based on the reduction approach, as well as the explanation; particularly good reductions may receive extra credit.




Notes:
------------------

(For general notes, see "Assignment Notes" section at the bottom of the assignment.)

	- The atomicMax function we give you has a return value, but you can ignore it - the function automatically stores the correct result in the memory location pointed to by the first argument.

	- In the "TODO 2" that asks you to allocate memory to store the maximum magnitude, allocate it to the previously-declared pointer "dev_max_abs_val".
	
	- For normalization, I mention that you have to normalize by the maximal magnitude of the signal. Since we're convolving two real signals, the output will be real (i.e. the complex component will be 0, modulo floating-point error). "Magnitude" then means the absolute value of the un-normalized output signal's real component. 

	(In theory, calculating the complex magnitude will get you this as well, but it is a bit inefficient.)




Hints:
------------------

A recommended resource is the presentation "Optimizing Parallel Reduction in CUDA", by Mark Harris.


Warning: If on the kernel call you pass in some amount of shared memory dynamically, and it's too 
large (like if you use padded_length as the amount of shared memory) you run the risk of 
overwriting the instruction registers and not being able to actually debug or run the kernel. 
(Of course Cuda doesn't seem to complain about this or throw an error.)


Assignment Notes
--------------------------------------------------------
--------------------------------------------------------

Just like in Homework 1, the code given to you will run the ordinary CPU version of the (naive algorithm) convolution, and check and the correctness of the GPU output. For large impulse responses, you will likely need to disable this (otherwise, computation will take a very long time).

As with Set 1, please make sure that your code is resilient to total number of threads being less than the data size.

There are two modes of operation:

	Normal mode: Generate the input signal x[n] randomly, with a size specified in the arguments, and use a Gaussian impulse response h[n].

	Audio mode: Read *both* the input signal x[n] and the impulse response h[n] from audio files, and write the signal y[n] as an output audio file. 


As in Homework 1, to toggle between these two modes, set AUDIO_ON accordingly, and use the appropriate makefile.

As before, Normal mode works on the servers haru, mx, and minuteman. 
Audio mode works only on haru.


We saw in class that various acoustic settings (e.g. rooms with lots of echo) can be modeled as LTI systems, and have impulse response. The assignment includes a sample input file ("example_testfile.wav"), as well as the impulse response of two different rooms (a truncated "silo" impulse, and that of a five-column room). Hence, convolving the original signal produces the audio "as played in these rooms"!

(Credit due: These impulse responses are from Voxengo (http://www.voxengo.com/impulses/) - you can download more from this website.)
