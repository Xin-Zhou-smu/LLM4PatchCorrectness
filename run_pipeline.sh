for tool in 'SOFix' 'ACS' 'Arja' 'AVATAR' 'CapGen'  'Cardumen'  'DynaMoth' 'FixMiner' 'GenProg' 'HDRepair'  'Jaid'  'jGenProg'  'jKali' 'jMutRepair' 'Kali'  'kPAR' 'Nopol' 'RSRepair' 'SequenceR' 'SimFix' 'SketchFix' 'TBar'

do
  for maxlength in 4000
  do

    for topk in 10
    do
        for option in  "bug-trace-testcase-coverage-similar"
        do
                                    echo ${tool}
                                    CUDA_VISIBLE_DEVICES=3 python main.py --task patch_${tool} --data_dir data_checked --split test \
                                        --out_dir checked_data_cross_tool_enhanced_results_starcoder_7b  \
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
