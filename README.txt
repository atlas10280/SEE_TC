System for Enhanced Evaluation of Tumor Cells (SEE-TC) is built to 
process multi-channel, whole-slide images of isolated and immunofluorescently 
labled tumor cells. It currently have several assumptions regarding data
format that are baked in as a result of being trained on images aquired using 
.nd2 image formats. The .nd2 document metadata is called upon several times to
automatically define critical parameters such as the zoom (default is 10x), 
channel names, or even the instrument used (if applying flat-field corrections). 
In development, the Hoechst stain was always on a 300nm wavelength channel. As
such, steps that utilize the Hoechst stain (segmentation, single-cell isolation)
expect the channel name to begin with '3'. This will need to be adjusted if 
your protocol differs.