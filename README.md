# ROMO: Retrieval-enhanced Offline Model-based Optimization
A `PyTorch` implementation of our DAI 2023 paper:
[ROMO: Retrieval-enhanced Offline Model-based Optimization](https://arxiv.org/abs/2310.07560).

If you have any questions, please contact the author: [Mingcheng Chen](http://apex.sjtu.edu.cn/members/cmc_iris@apexlab.org).

## Abstract
> Data-driven black-box model-based optimization (MBO) problems arise in a great number of practical application scenarios, where the goal is to find a design over the whole space maximizing a black-box target function based on a static offline dataset. In this work, we consider a more general but challenging MBO setting, named constrained MBO (CoMBO), where only part of the design space can be optimized while the rest is constrained by the environment. A new challenge arising from CoMBO is that most observed designs that satisfy the constraints are mediocre in evaluation. Therefore, we focus on optimizing these mediocre designs in the offline dataset while maintaining the given constraints rather than further boosting the best observed design in the traditional MBO setting. We propose retrieval-enhanced offline model-based optimization (ROMO), a new derivable forward approach that retrieves the offline dataset and aggregates relevant samples to provide a trusted prediction, and use it for gradient-based optimization. ROMO is simple to implement and outperforms state-of-the-art approaches in the CoMBO setting. Empirically, we conduct experiments on a synthetic Hartmann (3D) function dataset, an industrial CIO dataset, and a suite of modified tasks in the Design-Bench benchmark. Results show that ROMO performs well in a wide range of constrained optimization tasks.

## Citation
```
@misc{chen2023romo,
      title={ROMO: Retrieval-enhanced Offline Model-based Optimization}, 
      author={Mingcheng Chen and Haoran Zhao and Yuxiang Zhao and Hulei Fan and Hongqiao Gao and Yong Yu and Zheng Tian},
      year={2023},
      eprint={2310.07560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
