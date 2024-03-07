# LLM4PatchCorrectness


As the guthub has limititation on the file sizes, we use the zenodo to save the whole replication package. Please check this link (https://zenodo.org/record/7339088#.Y3oBonZBzIU) to download the complete repo.


## Pre-requirement
1. Python3.8+
2. CUDA Version: 11.7
3. Conda 

## Python Library Installation

```
$ conda create -n llm4correct python=3.8
$ conda activate llm4correct
$ bash bash install_library.sh
```


## In-context learning Inference


Please run the following pipeline script:

```
$  bash run_pipeline.sh
```

Notes:
1. '--task' the format is Patch_{APR_Tool_Name} and it is to choose the target APR tool, e.g. Patch_ACS 
2. When we submit to ICSE23, there are 'bigscience/bloom-1b3' and 'bigscience/bloom-1b1' available. While when writting the response, 'bigscience/bloom-1b3' is not available anymore. Thus, to run our code, you may need to use 'bigscience/bloom-1b1'. The results for each APR tools may diff, but the average performance of 'bigscience/bloom-1b1' and 'bigscience/bloom-1b3' are similar.
3. the code does not save the Bloom model as its model size is very huge; the content in the '--out_dir' is the logits generated by bloom models instead of a model checkpoint.
4. the default '--max_length' is 1000 while if you meet OOM problem, you can reduce it into 768.
5. the default '--batch_size' is 1 while if you have extra memory, you can set as 2 to speed up.


## Read Experiment Results

After finishing all inferences, you can run this python file to read the results for each APR tool:

```
$ python read_results.py
```

Notes:
1. Need to revised the path to the '--out_dir' in the last step






This also includes implementations of many recent papers studying in-context learning. 
* Brown et al. NeurIPS 2021. "[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)"
* Zhao et al. ICML 2021. "[Calibrate before use: Improving few-shot performance of language models](https://arxiv.org/abs/2102.09690)"
* Holzman et al. EMNLP 2021. "[Surface Form Competition: Why the Highest Probability Answer Isn't Always Right](https://arxiv.org/abs/2104.08315)"
* Sewon et al. 2022. "[Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/pdf/2108.04106.pdf)"



