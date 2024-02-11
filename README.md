# CAS: A Probability-based Approach for Universal Condition Alignment Score

>This repository contains the official implementation for the paper [CAS: A Probability-based Approach for Universal Condition Alignment Score](https://openreview.net/forum?id=E78OaH2s3f) <br> by [Chunsan Hong](https://sites.google.com/view/chunsanhong)<sup>*</sup>, Byunghee Cha<sup>*</sup>, Tae-Hyun Oh.

We propose a universal condition alignment score that leverages the conditional probability measurable through the diffusion process. Our technique operates across all conditions and requires no additional models beyond the diffusion model used for generation, effectively enabling self-rejection. Our experiments validate that our metric effectively applies in diverse conditional generations, such as text-to-image, {instruction, image}-to-image, edge-/scribble-to-image, and text-to-audio. The basic idea is captured in the figure below:

![Method Overview](assets/figures/mthd.png)

Examples of generated samples ordered by our universal score for various conditions is shown in the figure below:

![Generated Samples Examples](assets/figures/concept_ex.png)

From left to right, the sequence shows the conditional diffusion model used, the employed condition, two high-ranked samples, and two low-ranked samples. The last column’s generation is audio, so it is visualized using a Mel Spectrogram. Our metric demonstrates automatic and reliable assessment of perceptual alignment between condition and generated results (i.e., cherry-picking without humans) across text-to-image models (Stable Diffusion V1.5), diffusion models trained on specific domains such as Van Gogh, InstructPix2Pix, ControlNet, and AudioLDM, but not limited to.

## How to run the code
### Dependencies
Run the following to install a subset of necessary python packages for our code
```
pip install -r requirements.txt
```

## References
If you find the code useful for your research, please consider citing
```
@inproceedings{
  hong2024CAS,
  title={CAS: A Probability-based Approach for Universal Condition Alignment Score},
  author={Chunsan Hong and Byunghee Cha and Tae-Hyun Oh},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=E78OaH2s3f}
}
```