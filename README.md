# NeuroMamba: Behavior Score Prediction in Resting-State Functional MRI

![model](figures/model.png?raw=true)

Early clinical assessment of Alzheimer’s disease relies on behavior scores that measure a subject's language, memory, and cognitive skills. On the medical imaging side, functional magnetic resonance imaging has provided invaluable insights into the neural pathways underlying Alzheimer's disease. While prior studies have used resting-state functional MRI by extracting functional connectivity matrices, these approaches neglect the temporal dynamics inherent in functional data. In this work, we present a deep state space modeling framework that directly leverages the blood-oxygenation-level-dependent time series to learn a sparse collection of brain regions to predict behavior scores. Our model extracts temporal features that encapsulate nuanced patterns of intrinsic brain activity, thereby enhancing predictive performance compared to traditional connectivity methods. We identify specific brain regions that are most predictive of cognitive impairment through experiments on data provided by the Michigan Alzheimer's Disease Research Center, providing new insights into the neural substrates of early Alzheimer's pathology. These findings have important implications for the possible development of risk monitoring and intervention strategies in Alzheimer’s disease. 

# Table Of Contents
-  [Directory Structure](#directory-structure)
-  [Installation](#installation)
-  [Fmriprep preprocessing](#Preprocessing-fmriprep)
-  [Conn toolbox preprocessing](#Preprocessing-CONNToolbox)
-  [Paper Citation](#Citation)

# Directory Structure
```
├──  experiments - training and evaluation scripts for various methods
     ├──  ablation - comparing neuromamba to base mamba model
     ├──  adni - zero-shot transfer, few shot tuning, and all-shot training on ADNI data
     ├──  crossentropy - classification variant to see if rsfMRI + MoCA is useful for diagnosis
     ├──  importance - use PFI to find which ROIs were most useful for diagnosis
     ├──  leaveoneout - main pearson correlation results using LOO testing methodology
     ├──  madc - generate statistics and violin plot of data distribution etc...
     ├──  roc - code to generate roc curves given tested methods
     ├──  saliency - saliency plots supporting spontaneous activity in DMN
     ├──  scatter - plot scatter plots of trained neuromamba model showing true vs predicted scores
     ├──  validation - quick check of model hyperparamaters by 2-fold testing
├──  figures - generated plots/results shown in the paper
     ├── importance_language.tex - importance result for language metric
     ├── importance_memory.tex - importance result for memory metric
     ├── importance_moca.tex - importance result for moca metric
     ├── madc_violin.pdf - violin plot of MADRC data distribution
     ├── model.png/model.pdf - neuromamba model illustration in paper
     ├── roc.pdf - roc curve with AUC values for a few methods
     ├── saliency.pdf - saliency maps of neuromamba model showing activity in DMN
     ├── scatter_language.pdf - scatter plot for language metric across subjects
     ├── scatter_memory.pdf - scatter plot for memory metric across subjects
     ├── scatter_moca.pdf - scatter plot for moca metric across subjects
├──  models - classical model-based and data-driven deep methods used in the paper
     ├── ALFF.py - amplitude of low frequency fluctuations
     ├── FCM.py - generate functional connectivity matrices given timeseries data
     ├── GICA.py - group ica method outlined in paper
     ├── IICA.py - individual ica method outlined in paper
     ├── LSTM.py - long short term memory networks
     ├── NeuroMamba.py - our method as shown in figure above
     ├── PatchTST.py - patch timeseries transformer method
     ├── TCN.py - temporal convolutional network method
├──  scripts - some collection of python scripts needed to do offline processing
     ├── adni_moca.csv - csv provided by ADNI group that contains raw MoCA scores for ADNI data
     ├── conn_toolbox.py - code to convert the .mat ROI struct files from MATLAB to normal matrices
     ├── exam_sessions.csv - list of exam sessions to figure out which MADRC data entries correspond to the mri date
     ├── generate_labels.py - code to give diagnosis labels (cn,amci,dat) to subjects and filter for coditions such as lewy body
     ├── generate_zscores.py - convert raw metric to normative z-score space based on healty population
     ├── map_exam_sessions.py - code used in conjunction with exam_sessions.csv to find closest MADRC entry that matches MRI date
     ├── prep_adni.py - code that converts adni_moca.csv to adni.csv used in this project, labels included etc...
     ├── regress_variates.py - code to remove effects of age, education, sex, and race on MoCA scores
├──  utils - dataloaders, loss functions, metrics, functional connectivity generation etc...
     ├── adni.py - dataloaders and functions used only for ADNI data
     ├── dataloaders.py - dataset definitions and dataloaders for MADRC data
     ├── fmri.py - some fmri-specific functions  
├──  weights - a symbolic link to another location on my drive due to github space limitations
     ├── if you would like access to model weights, please contact Javier or Scott (see emails in paper)
├── adni.csv - database of adni files and corresponding metrics/labels
├── madc_complete.csv - database of entire MADRC data with full columns and info
├── power_atlas.csv - Jonathan Power's atlas of ROIs used to identify relevant brain regions
├── readme.md - file you are looking at right now
├── scores.csv - condenced MADRC database with processed columns used in this project
```

# Installation
1. Append this to ~/.bashrc so that package is exposed, required to use lines such as "import NeuroMamba.models.FCM" used in basically all python files for this project. Replace directory with wherever you git clone this repo, e.g., /home/javier/Desktop in my computer.
```
export PYTHONPATH="${PYTHONPATH}:/home/javier/Desktop"
```

2. Create conda environment
```
conda create -n deepscore python=3.12
conda activate deepscore
```

3. Install the following packages listed below. I have an AMD GPU so pytorch's rocm branch is used, replace as needed. Because of this, I must compile mamba-ssm and causal-conv1d wheels for rocm manually. This was tested using Python 3.12 and Pytorch 2.9 and ROCM 6.4 on Fedora Linux 43 on kernel 6.18.3 near the end of project submission.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
git clone https://github.com/Dao-AILab/causal-conv1d && cd causal-conv1d && pip install . --no-build-isolation && cd ~
git clone https://github.com/state-spaces/mamba && cd ./mamba && pip install . --no-build-isolation && cd ~
pip install torchinfo
pip install scipy
pip install scikit-learn
pip install pandas
pip install seaborn[stats]
```

# Preprocessing-fmriprep

Results included in this manuscript come from preprocessing
performed using **fMRIPrep** 21.0.3
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on **Nipype** 1.6.1
(@nipype1; @nipype2; RRID:SCR_002502).
Many internal operations of **fMRIPrep** use
**Nilearn** 0.8.1 [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in **fMRIPrep** documentation](https://fmriprep.readthedocs.io/en/latest/workflows.html "FMRIPrep's documentation").

## Methods (fmriprep)

**Preprocessing of B0 inhomogeneity mappings:**
A total of 2 fieldmaps were found available within the input
BIDS structure for this particular subject.
A **B0**-nonuniformity map (or **fieldmap**) was estimated based on two (or more)
echo-planar imaging (EPI) references  with `topup` (@topup; FSL 6.0.5.1:57b01774).

**Anatomical data preprocessing:**
A total of 1 T1-weighted (T1w) images were found within the input
BIDS dataset.The T1-weighted (T1w) image was corrected for intensity non-uniformity (INU)
with `N4BiasFieldCorrection` [@n4], distributed with ANTs 2.3.3 [@ants, RRID:SCR_004757], and used as T1w-reference throughout the workflow.
The T1w-reference was then skull-stripped with a **Nipype** implementation of
the `antsBrainExtraction.sh` workflow (from ANTs), using OASIS30ANTs
as target template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL 6.0.5.1:57b01774, RRID:SCR_002823,
@fsl_fast].
Brain surfaces were reconstructed using `recon-all` [FreeSurfer 6.0.1,
RRID:SCR_001847, @fs_reconall], and the brain mask estimated
previously was refined with a custom variation of the method to reconcile
ANTs-derived and FreeSurfer-derived segmentations of the cortical
gray-matter of Mindboggle [RRID:SCR_002438, @mindboggle].
Volume-based spatial normalization to two standard spaces (MNI152NLin6Asym, MNI152NLin2009cAsym) was performed through
nonlinear registration with `antsRegistration` (ANTs 2.3.3),
using brain-extracted versions of both T1w reference and the T1w template.
The following templates were selected for spatial normalization:
**FSL's MNI ICBM 152 non-linear 6th Generation Asymmetric Average Brain Stereotaxic Registration Model** [@mni152nlin6asym, RRID:SCR_002823; TemplateFlow ID: MNI152NLin6Asym], **ICBM 152 Nonlinear Asymmetrical template version 2009c** [@mni152nlin2009casym, RRID:SCR_008796; TemplateFlow ID: MNI152NLin2009cAsym].

**Functional data preprocessing:**
For each of the 2 BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
First, a reference volume and its skull-stripped version were generated using a custom
methodology of **fMRIPrep**. Head-motion parameters with respect to the BOLD reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
`mcflirt` [FSL 6.0.5.1:57b01774, @mcflirt].
The estimated **fieldmap** was then aligned with rigid-registration to the target
EPI (echo-planar imaging) reference run.
The field coefficients were mapped on to the reference EPI using the transform.
The BOLD reference was then co-registered to the T1w reference using
`bbregister` (FreeSurfer) which implements boundary-based registration [@bbr].
Co-registration was configured with six degrees of freedom.
Several confounding time-series were calculated based on the
**preprocessed BOLD**: framewise displacement (FD), DVARS and
three region-wise global signals.
FD was computed using two formulations following Power (absolute sum of
relative motions, @power_fd_dvars) and Jenkinson (relative root mean square
displacement between affines, @mcflirt).
FD and DVARS are calculated for each functional run, both using their
implementations in **Nipype** [following the definitions by @power_fd_dvars].
The three global signals are extracted within the CSF, the WM, and
the whole-brain masks.
Additionally, a set of physiological regressors were extracted to
allow for component-based noise correction [**CompCor**, @compcor].
Principal components are estimated after high-pass filtering the
**preprocessed BOLD** time-series (using a discrete cosine filter with
128s cut-off) for the two **CompCor** variants: temporal (tCompCor)
and anatomical (aCompCor).
tCompCor components are then calculated from the top 2% variable
voxels within the brain mask.
For aCompCor, three probabilistic masks (CSF, WM and combined CSF+WM)
are generated in anatomical space.
The implementation differs from that of Behzadi et al. in that instead
of eroding the masks by 2 pixels on BOLD space, the aCompCor masks are
subtracted a mask of pixels that likely contain a volume fraction of GM.
This mask is obtained by dilating a GM mask extracted from the FreeSurfer's **aseg** segmentation, and it ensures components are not extracted from voxels containing a minimal fraction of GM.
Finally, these masks are resampled into BOLD space and binarized by
thresholding at 0.99 (as in the original implementation).
Components are also calculated separately within the WM and CSF masks.
For each CompCor decomposition, the **k** components with the largest singular
values are retained, such that the retained components' time series are
sufficient to explain 50 percent of variance across the nuisance mask (CSF,
WM, combined, or temporal). The remaining components are dropped from
consideration.
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file.
The confound time series derived from head motion estimates and global
signals were expanded with the inclusion of temporal derivatives and
quadratic terms for each [@confounds_satterthwaite_2013].
Frames that exceeded a threshold of 0.5 mm FD or
1.5 standardised DVARS were annotated as motion outliers.
The BOLD time-series were resampled into standard space,
generating a **preprocessed BOLD run in MNI152NLin6Asym space**.
First, a reference volume and its skull-stripped version were generated
using a custom methodology of **fMRIPrep**.
The BOLD time-series were resampled onto the following surfaces
(FreeSurfer reconstruction nomenclature):
**fsaverage**.
Automatic removal of motion artifacts using independent component analysis
[ICA-AROMA, @aroma] was performed on the **preprocessed BOLD on MNI space**
time-series after removal of non-steady state volumes and spatial smoothing
with an isotropic, Gaussian kernel of 6mm FWHM (full-width half-maximum).
Corresponding "non-aggresively" denoised runs were produced after such
smoothing.
Additionally, the "aggressive" noise-regressors were collected and placed
in the corresponding confounds file.
**Grayordinates** files [@hcppipelines] containing 91k samples were also
generated using the highest-resolution ``fsaverage`` as intermediate standardized
surface space.
All resamplings can be performed with **a single interpolation step** by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).

## References (fmriprep)
Abraham, Alexandre, Fabian Pedregosa, Michael Eickenberg, Philippe Gervais, Andreas Mueller, Jean Kossaifi, Alexandre Gramfort, Bertrand Thirion, and Gael Varoquaux. 2014. “Machine Learning for Neuroimaging with Scikit-Learn.” Frontiers in Neuroinformatics 8. https://doi.org/10.3389/fninf.2014.00014.

Andersson, Jesper L. R., Stefan Skare, and John Ashburner. 2003. “How to Correct Susceptibility Distortions in Spin-Echo Echo-Planar Images: Application to Diffusion Tensor Imaging.” NeuroImage 20 (2): 870–88. https://doi.org/10.1016/S1053-8119(03)00336-7.

Avants, B. B., C. L. Epstein, M. Grossman, and J. C. Gee. 2008. “Symmetric Diffeomorphic Image Registration with Cross-Correlation: Evaluating Automated Labeling of Elderly and Neurodegenerative Brain.” Medical Image Analysis 12 (1): 26–41. https://doi.org/10.1016/j.media.2007.06.004.

Behzadi, Yashar, Khaled Restom, Joy Liau, and Thomas T. Liu. 2007. “A Component Based Noise Correction Method (CompCor) for BOLD and Perfusion Based fMRI.” NeuroImage 37 (1): 90–101. https://doi.org/10.1016/j.neuroimage.2007.04.042.

Dale, Anders M., Bruce Fischl, and Martin I. Sereno. 1999. “Cortical Surface-Based Analysis: I. Segmentation and Surface Reconstruction.” NeuroImage 9 (2): 179–94. https://doi.org/10.1006/nimg.1998.0395.

Esteban, Oscar, Ross Blair, Christopher J. Markiewicz, Shoshana L. Berleant, Craig Moodie, Feilong Ma, Ayse Ilkay Isik, et al. 2018. “fMRIPrep.” Software. https://doi.org/10.5281/zenodo.852659.

Esteban, Oscar, Christopher Markiewicz, Ross W Blair, Craig Moodie, Ayse Ilkay Isik, Asier Erramuzpe Aliaga, James Kent, et al. 2018. “fMRIPrep: A Robust Preprocessing Pipeline for Functional MRI.” Nature Methods. https://doi.org/10.1038/s41592-018-0235-4.

Evans, AC, AL Janke, DL Collins, and S Baillet. 2012. “Brain Templates and Atlases.” NeuroImage 62 (2): 911–22. https://doi.org/10.1016/j.neuroimage.2012.01.024.

Fonov, VS, AC Evans, RC McKinstry, CR Almli, and DL Collins. 2009. “Unbiased Nonlinear Average Age-Appropriate Brain Templates from Birth to Adulthood.” NeuroImage 47, Supplement 1: S102. https://doi.org/10.1016/S1053-8119(09)70884-5.

Glasser, Matthew F., Stamatios N. Sotiropoulos, J. Anthony Wilson, Timothy S. Coalson, Bruce Fischl, Jesper L. Andersson, Junqian Xu, et al. 2013. “The Minimal Preprocessing Pipelines for the Human Connectome Project.” NeuroImage, Mapping the connectome, 80: 105–24. https://doi.org/10.1016/j.neuroimage.2013.04.127.

Gorgolewski, K., C. D. Burns, C. Madison, D. Clark, Y. O. Halchenko, M. L. Waskom, and S. Ghosh. 2011. “Nipype: A Flexible, Lightweight and Extensible Neuroimaging Data Processing Framework in Python.” Frontiers in Neuroinformatics 5: 13. https://doi.org/10.3389/fninf.2011.00013.

Gorgolewski, Krzysztof J., Oscar Esteban, Christopher J. Markiewicz, Erik Ziegler, David Gage Ellis, Michael Philipp Notter, Dorota Jarecka, et al. 2018. “Nipype.” Software. https://doi.org/10.5281/zenodo.596855.

Greve, Douglas N, and Bruce Fischl. 2009. “Accurate and Robust Brain Image Alignment Using Boundary-Based Registration.” NeuroImage 48 (1): 63–72. https://doi.org/10.1016/j.neuroimage.2009.06.060.

Jenkinson, Mark, Peter Bannister, Michael Brady, and Stephen Smith. 2002. “Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images.” NeuroImage 17 (2): 825–41. https://doi.org/10.1006/nimg.2002.1132.

Klein, Arno, Satrajit S. Ghosh, Forrest S. Bao, Joachim Giard, Yrjö Häme, Eliezer Stavsky, Noah Lee, et al. 2017. “Mindboggling Morphometry of Human Brains.” PLOS Computational Biology 13 (2): e1005350. https://doi.org/10.1371/journal.pcbi.1005350.

Lanczos, C. 1964. “Evaluation of Noisy Data.” Journal of the Society for Industrial and Applied Mathematics Series B Numerical Analysis 1 (1): 76–85. https://doi.org/10.1137/0701007.

Power, Jonathan D., Anish Mitra, Timothy O. Laumann, Abraham Z. Snyder, Bradley L. Schlaggar, and Steven E. Petersen. 2014. “Methods to Detect, Characterize, and Remove Motion Artifact in Resting State fMRI.” NeuroImage 84 (Supplement C): 320–41. https://doi.org/10.1016/j.neuroimage.2013.08.048.

Pruim, Raimon H. R., Maarten Mennes, Daan van Rooij, Alberto Llera, Jan K. Buitelaar, and Christian F. Beckmann. 2015. “ICA-AROMA: A Robust ICA-Based Strategy for Removing Motion Artifacts from fMRI Data.” NeuroImage 112 (Supplement C): 267–77. https://doi.org/10.1016/j.neuroimage.2015.02.064.

Satterthwaite, Theodore D., Mark A. Elliott, Raphael T. Gerraty, Kosha Ruparel, James Loughead, Monica E. Calkins, Simon B. Eickhoff, et al. 2013. “An improved framework for confound regression and filtering for control of motion artifact in the preprocessing of resting-state functional connectivity data.” NeuroImage 64 (1): 240–56. https://doi.org/10.1016/j.neuroimage.2012.08.052.

Tustison, N. J., B. B. Avants, P. A. Cook, Y. Zheng, A. Egan, P. A. Yushkevich, and J. C. Gee. 2010. “N4itk: Improved N3 Bias Correction.” IEEE Transactions on Medical Imaging 29 (6): 1310–20. https://doi.org/10.1109/TMI.2010.2046908.

Zhang, Y., M. Brady, and S. Smith. 2001. “Segmentation of Brain MR Images Through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm.” IEEE Transactions on Medical Imaging 20 (1): 45–57. https://doi.org/10.1109/42.906424.


# Preprocessing-CONNToolbox

## Methods (CONN Toolbox)

Analyses of fMRI data were performed using CONN<sup>\[1\]</sup> (RRID:SCR_009550) release 22.v2407<sup>\[2\]</sup> and SPM<sup>\[3\]</sup> (RRID:SCR_007037) release 12.7771.

**Preprocessing**: Functional and anatomical data were preprocessed using a modular preprocessing pipeline<sup>\[4\]</sup> including removal of initial scans, realignment, outlier detection, and smoothing. The first 12 scans in each functional run were removed. Functional data were coregistered to a reference image (first scan of the first session) using a least squares approach and a 6 parameter (rigid body) transformation without resampling<sup>\[5\]</sup>. Potential outlier scans were identified using ART<sup>\[6\]</sup> as acquisitions with framewise displacement above 0.5 mm or global BOLD signal changes above 3 standard deviations<sup>\[7,8\]</sup>, and a reference BOLD image was computed for each subject by averaging all scans excluding outliers. Last, functional data were smoothed using spatial convolution with a Gaussian kernel of 6 mm full width half maximum (FWHM).

**Denoising**: In addition, functional data were denoised using a standard denoising pipeline<sup>\[9\]</sup> including the regression of potential confounding effects characterized by white matter timeseries (5 CompCor noise components), CSF timeseries (5 CompCor noise components), motion parameters (6 factors)<sup>\[10\]</sup>, outlier scans (below 100 factors)<sup>\[7\]</sup>, session effects and their first order derivatives (2 factors), and linear trends (2 factors) within each functional run, followed by bandpass frequency filtering of the BOLD timeseries<sup>\[11\]</sup> between 0.008 Hz and 0.09 Hz. CompCor<sup>\[12,13\]</sup> noise components within white matter and CSF were estimated by computing the average BOLD signal as well as the largest principal components orthogonal to the BOLD average, motion parameters, and outlier scans within each subject's eroded segmentation masks. From the number of noise terms included in this denoising strategy, the effective degrees of freedom of the BOLD signal after denoising were estimated to range from 59 to 72.2 (average 70.1) across all subjects<sup>\[8\]</sup>.

**First-level analysis** SBC_01: Seed-based connectivity maps (SBC) and ROI-to-ROI connectivity matrices (RRC) were estimated characterizing the patterns of functional connectivity with 272 ROIs. Functional connectivity strength was represented by Fisher-transformed bivariate correlation coefficients from a weighted general linear model (weighted-GLM<sup>\[14\]</sup>), defined separately for each pair of seed and target areas, modeling the association between their BOLD signal timeseries. In order to compensate for possible transient magnetization effects at the beginning of each run, individual scans were weighted by a step function convolved with an SPM canonical hemodynamic response function and rectified.

## References (CONN Toolbox)

<sup>\[1\]</sup> Whitfield-Gabrieli, S., & Nieto-Castanon, A. (2012). Conn: a functional connectivity toolbox for correlated and anticorrelated brain networks. Brain connectivity, 2(3), 125-141.

<sup>\[2\]</sup> Nieto-Castanon, A. & Whitfield-Gabrieli, S. (2022). CONN functional connectivity toolbox: RRID SCR_009550, release 22. doi:10.56441/hilbertpress.2246.5840.

<sup>\[3\]</sup> Penny, W. D., Friston, K. J., Ashburner, J. T., Kiebel, S. J., & Nichols, T. E. (Eds.). (2011). Statistical parametric mapping: the analysis of functional brain images. Elsevier.

<sup>\[4\]</sup> Nieto-Castanon, A. (2020). FMRI minimal preprocessing pipeline. In Handbook of functional connectivity Magnetic Resonance Imaging methods in CONN (pp. 3-16). Hilbert Press.

<sup>\[5\]</sup> Friston, K. J., Ashburner, J., Frith, C. D., Poline, J. B., Heather, J. D., & Frackowiak, R. S. (1995). Spatial registration and normalization of images. Human brain mapping, 3(3), 165-189.

<sup>\[6\]</sup> Whitfield-Gabrieli, S., Nieto-Castanon, A., & Ghosh, S. (2011). Artifact detection tools (ART). Cambridge, MA. Release Version, 7(19), 11.

<sup>\[7\]</sup> Power, J. D., Mitra, A., Laumann, T. O., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2014). Methods to detect, characterize, and remove motion artifact in resting state fMRI. Neuroimage, 84, 320-341.

<sup>\[8\]</sup> Nieto-Castanon, A. (submitted). Preparing fMRI Data for Statistical Analysis. In M. Filippi (Ed.). fMRI techniques and protocols. Springer. doi:10.48550/arXiv.2210.13564

<sup>\[9\]</sup> Nieto-Castanon, A. (2020). FMRI denoising pipeline. In Handbook of functional connectivity Magnetic Resonance Imaging methods in CONN (pp. 17-25). Hilbert Press.

<sup>\[10\]</sup> Friston, K. J., Williams, S., Howard, R., Frackowiak, R. S., & Turner, R. (1996). Movement-related effects in fMRI time-series. Magnetic resonance in medicine, 35(3), 346-355.

<sup>\[11\]</sup> Hallquist, M. N., Hwang, K., & Luna, B. (2013). The nuisance of nuisance regression: spectral misspecification in a common approach to resting-state fMRI preprocessing reintroduces noise and obscures functional connectivity. Neuroimage, 82, 208-225.

<sup>\[12\]</sup> Behzadi, Y., Restom, K., Liau, J., & Liu, T. T. (2007). A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. Neuroimage, 37(1), 90-101.

<sup>\[13\]</sup> Chai, X. J., Nieto-Castanon, A., Ongur, D., & Whitfield-Gabrieli, S. (2012). Anticorrelations in resting state networks without global signal regression. Neuroimage, 59(2), 1420-1428.

<sup>\[14\]</sup> Nieto-Castanon, A. (2020). Functional Connectivity measures. In Handbook of functional connectivity Magnetic Resonance Imaging methods in CONN (pp. 26-62). Hilbert Press.

# Citation
Pending review at IEEE Transactions on Computational Biology and Bioinformatics.
Arxiv citation below:
```
@article{cavazos2026behavior,
  title={Behavior Score Prediction in Resting-State Functional MRI by Deep State Space Modeling},
  author={Cavazos, Javier Salazar and Egan, Maximillian and Litinas, Krisanne and Hampstead, Benjamin and Peltier, Scott},
  journal={arXiv preprint arXiv:2602.07131},
  year={2026}
}
```





