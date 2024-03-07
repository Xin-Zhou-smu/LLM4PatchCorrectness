# LLM4PatchCorrectness


As the guthub has limititation on the file sizes, we use the zenodo to save the whole replication package. Please check this link (https://zenodo.org/record/7339088#.Y3oBonZBzIU) to download the complete repo.


## Pre-requirement
1. Python3.8+
2. CUDA Version: 11.7
3. Conda 

Note: The CUDA Version needs to be 11.7 to ensure compatibility and functionality.

## Python Library Installation

```
$ conda create -n llm4correct python=3.8
$ conda activate llm4correct

This step may take several minutes: 
$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

$ bash install_library.sh
```


## In-context learning Inference


Please run the following pipeline script:

```
$  bash run_pipeline.sh
```

Notes:
1. '--task' the format is Patch_{APR_Tool_Name} and it is to choose the target APR tool, e.g. Patch_ACS
2. '--option' is to choose the guiding information for LLM. "bug-trace-testcase-similar" is the default parameter. 
3. the content in the '--out_dir' is the logits generated by LLM.
4. the default '--max_length' is 4000 while if you meet OOM problem, you can reduce it accordingly.
5. the default '--batch_size' is 1 while if you have extra memory, you can set to 2 to speed up.


## Read Experiment Results

After finishing all inferences, you can run this Python file to read the results for each APR tool:

```
$ python read_results.py
```

Notes:
1. Need to revise the path to the '--out_dir' in the last step




This also includes implementations of many recent papers studying in-context learning. 
* Brown et al. NeurIPS 2021. "[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)"
* Zhao et al. ICML 2021. "[Calibrate before use: Improving few-shot performance of language models](https://arxiv.org/abs/2102.09690)"
* Holzman et al. EMNLP 2021. "[Surface Form Competition: Why the Highest Probability Answer Isn't Always Right](https://arxiv.org/abs/2104.08315)"
* Sewon et al. 2022. "[Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/pdf/2108.04106.pdf)"



