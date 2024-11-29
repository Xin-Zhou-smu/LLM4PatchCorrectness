for tool in 'ACS' 'Arja' 'AVATAR' 'CapGen'  'Cardumen' 'DynaMoth' 'FixMiner'  'GenProg' 'HDRepair' 'Jaid' 'jGenProg' 'jKali' 'jMutRepair' 'Kali' 'kPAR' 'Nopol' 'RSRepair' 'SequenceR' 'SimFix' 'SketchFix' 'SOFix' 'TBar'
do
  for maxlength in 4000
  do

    for topk in 10
    do
        for option in  "bug-trace-testcase-coverage-similar"
        do
                                
                         
                                    echo ${tool}
                                    CUDA_VISIBLE_DEVICES=1 python main.py --task patch_${tool} --data_dir data_checked --split test \
                                        --out_dir results  \
                                        --gpt2  bigcode/starcoderbase-7b \
                                        --batch_size 1  --do_zeroshot --k 10 \
                                        --max_length ${maxlength} \
                                        --n_template 0 \
                                        --top_k_example ${topk} \
                                        --sim_threshold 0.9 \
                                        --enhancement_option  ${option}
        done
    done
  done
done
