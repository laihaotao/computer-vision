Implementation for part 1:
Step 0: read images into cv::Mat and use split function to split 3 channels;
Step 1: Use loops to extract R,G,B channels and fill the "?"-place to 0 respectively,
        coded in functions fillZero4BlueChannel, fillZero4GreenChannel and fillZero4RedChannel,
        details commented in the cpp file.
Step 2: Use 2 kernels to filter blue and red channel, use 1 kernel to filter the green channel;
Step 3: merge the 3 filtered channel and get the demosaic picture.



Implementation for part 2:
commented in cpp code - function: freemanImprvDemosaic.



Answer to part 1 question:
As the screen shots show (named part_oldwell_origin.png & part_oldwell_demosaic.png in the resources
folder). The demosaic one shows blurring on edges. Especially in red and blue.
That's because the "?" - places are estimated by averaging some pixels around. And the number of
red and blue pixels are much less than green.



Answer to part 2 question:
Yes, there're visible improvements. Because green channel is more sampled than blue and red, which
makes green channel performs better. This algorithm utilize the performance of green channel, so the
blurring reduced.



Difficulties I met:
0. I don't know C++ at all, so stuck for long in pointers especially when extracting and modifying the values in cv::Mat
1. code style seems ugly.
