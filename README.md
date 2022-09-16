# PL-GAN Models
[IEEE ](https://ieeexplore.ieee.org/document/9866771) | [BibTeX](#bibtex)

<p align="center">
<img src=assets/results.gif />
</p>



[**PL-GAN: Path Loss Prediction using Generative Adversarial Networks**](https://ieeexplore.ieee.org/document/9866771)<br/>
[Ahmed Marey](https://github.com/ahmarey),
Mustafa Bal,
[Hasan Ates](https://github.com/hfates)\,
[Bahadir Gunturk](https://github.com/bahadirgunturk)

This is a repository for the Path Loss GAN (PL-GAN) project. PL-GAN infers excess path loss of an area from a satellite or height map image.

Running the evaluation code infer.py will generate the path loss images and the statistical comparison between height map input and satellite image input.

The test set can be downloaded from [here](https://drive.google.com/drive/folders/1DgGqWcX1VYvIf8YDjjmr6WHAO1bJBvUN?usp=sharing). 
Place it in the test folder and run infer.py.

The figure below shows input images and results for some sample regions.

![](paper_table.png)


## BibTeX

```
@ARTICLE{9866771,
  author={Marey, Ahmed and Bal, Mustafa and Ates, Hasan F. and Gunturk, Bahadir K.},
  journal={IEEE Access}, 
  title={PL-GAN: Path Loss Prediction Using Generative Adversarial Networks}, 
  year={2022},
  volume={10},
  number={},
  pages={90474-90480},
  doi={10.1109/ACCESS.2022.3201643}}
```
