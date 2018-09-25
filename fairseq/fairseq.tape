package onmt
    :: .versioner=git .repo="https://github.com/SudaMonster/OpenNMT-py" .ref=HEAD
    :: realref="refs/remotes/origin/master" {

  git checkout -b working $realref
}

package subword_nmt :: .versioner=git .repo="https://github.com/rsennrich/subword-nmt" .ref=HEAD {

}

package mosesdecoder :: .versioner=git .repo="https://github.com/moses-smt/mosesdecoder" .ref=HEAD {

}

# there seems to be a bug in branching, so this is necessary
task tokenize : mosesdecoder
    < in=data
    > out
    :: lang=(side: src=$src tgt=$tgt)
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
        $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang < $in | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $lang > $out
    }


task train_truecaser : mosesdecoder
    < in=$out@tokenize[section:train]
    > out
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
        $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $in -model $out
    }


task truecase : mosesdecoder
    < in=$out@tokenize
    < model=$out@train_truecaser
    > out
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
    $mosesdecoder/scripts/recaser/truecase.perl -model $model < $in > $out
    }


task train_bpe : subword_nmt
    < in=$bpe_input[section:train]
    > out
    :: bpe_operations=@
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
      cat $in | $subword_nmt/learn_bpe.py -s $bpe_operations > $out
    }


task apply_bpe : subword_nmt
    < in=$bpe_input
    < model=$out@train_bpe
    > out 
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
      $subword_nmt/apply_bpe.py -c $model < $in > $out
    }

task clean_corpus : mosesdecoder
    < train_src_in=$out@apply_bpe[section:train,side:src]
    < train_tgt_in=$out@apply_bpe[section:train,side:tgt]
    > train_src_out
    > train_tgt_out
    :: src=@
    :: tgt=@
    :: clean_minlen=@
    :: clean_maxlen=@
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=""
    {
      
      ln -s $train_src_in train.$src
      ln -s $train_tgt_in train.$tgt
      $mosesdecoder/scripts/training/clean-corpus-n.perl ./train $src $tgt ./train.clean $clean_minlen $clean_maxlen
      ln -s ./train.clean.$src $train_src_out
      ln -s ./train.clean.$tgt $train_tgt_out
    }

task binarize_data : onmt mosesdecoder
    < train_src_in=$train_src_out@clean_corpus
    < train_tgt_in=$train_tgt_out@clean_corpus
    < dev_src_in=$out@apply_bpe[section:dev,side:src]
    < dev_tgt_in=$out@apply_bpe[section:dev,side:tgt]
    < test_src_in=$out@apply_bpe[section:test,side:src]
    < test_tgt_in=$out@apply_bpe[section:test,side:tgt]
    > out
    :: python_env=@
    :: src=@
    :: tgt=@
    :: .submitter=$submitter
    :: .resource_flags=$cpu_resource_flags 
    :: .action_flags=$action_flags
    {
      source activate $python_env
      
      ln -s $train_src_in train.bpe.$src
      ln -s $train_tgt_in train.bpe.$tgt
      ln -s $dev_src_in dev.bpe.$src
      ln -s $dev_tgt_in dev.bpe.$tgt
      ln -s $test_src_in test.bpe.$src
      ln -s $test_tgt_in test.bpe.$tgt


      python $onmt/preprocess.py \
        -train_src train.bpe.$src \
        -train_tgt train.bpe.$tgt \
        -valid_src dev.bpe.$src \
        -valid_tgt dev.bpe.$tgt \
        -save_data $out

      if [ -f ${out}.train.1.pt ] ; then
        touch $out
      fi
    }
#
task train : onmt
    < in=$out@binarize_data
    > out
    > train_log
    :: python_env=@
    :: save_checkpoint_steps=@
    :: keep_checkpoint=@
    :: seed=@
    :: param_init=@
    :: batch_size=@
    :: batch_type=@
    :: normalization=@
    :: valid_steps=@
    :: valid_batch_size=@
    :: max_generator_batches=@
    :: train_steps=@
    :: optim=@
    :: max_grad_norm=@
    :: dropout=@
    :: adam_beta1=@
    :: adam_beta2=@
    :: label_smoothing=@
    :: learning_rate=@
    :: learning_rate_decay=@
    :: start_decay_steps=@
    :: decay_steps=@
    :: report_every=@
    :: src_word_vec_size=@
    :: tgt_word_vec_size=@
    :: feat_merge=@
    :: feat_vec_size=@
    :: feat_vec_exponent=@
    :: encoder_type=@
    :: decoder_type=@
    :: enc_layers=@
    :: dec_layers=@
    :: rnn_size=@
    :: rnn_type=@
    :: global_attention=@
    :: global_attention_function=@
    :: self_attn_type=@
    :: generator_function=@
    :: .submitter=$submitter
    :: .resource_flags=$gpu_resource_flags 
    :: .action_flags=$action_flags
    {
        source activate $python_env
      
        cmd="python -u $onmt/train.py "
        cmd="$cmd -data $in "
        cmd="$cmd -save_model $out "
        cmd="$cmd -save_checkpoint_steps ${save_checkpoint_steps} "
        cmd="$cmd -keep_checkpoint ${keep_checkpoint} "
        cmd="$cmd -seed ${seed} "
        cmd="$cmd -gpuid 0"
        cmd="$cmd -param_init ${param_init} "
        cmd="$cmd -batch_size ${batch_size} "
        cmd="$cmd -batch_type ${batch_type} "
        cmd="$cmd -normalization ${normalization} "
        cmd="$cmd -valid_steps ${valid_steps} "
        cmd="$cmd -valid_batch_size ${valid_batch_size} "
        cmd="$cmd -max_generator_batches ${max_generator_batches} "
        cmd="$cmd -train_steps ${train_steps} "
        cmd="$cmd -optim ${optim} "
        cmd="$cmd -max_grad_norm ${max_grad_norm} "
        cmd="$cmd -dropout ${dropout} "
        cmd="$cmd -adam_beta1 ${adam_beta1} "
        cmd="$cmd -adam_beta2 ${adam_beta2} "
        cmd="$cmd -label_smoothing ${label_smoothing} "
        cmd="$cmd -learning_rate ${learning_rate} "
        cmd="$cmd -learning_rate_decay ${learning_rate_decay} "
        cmd="$cmd -start_decay_steps ${start_decay_steps} "
        cmd="$cmd -decay_steps ${decay_steps} "
        cmd="$cmd -report_every ${report_every}"
        cmd="$cmd -src_word_vec_size ${src_word_vec_size} "
        cmd="$cmd -tgt_word_vec_size ${tgt_word_vec_size} "
        cmd="$cmd -feat_merge ${feat_merge} "
        cmd="$cmd -feat_vec_size ${feat_vec_size} "
        cmd="$cmd -feat_vec_exponent ${feat_vec_exponent} "
        cmd="$cmd -encoder_type ${encoder_type} "
        cmd="$cmd -decoder_type ${decoder_type} "
        cmd="$cmd -enc_layers ${enc_layers} "
        cmd="$cmd -dec_layers ${dec_layers} "
        cmd="$cmd -rnn_size ${rnn_size} "
        cmd="$cmd -rnn_type ${rnn_type} "
        cmd="$cmd -global_attention ${global_attention}"
        cmd="$cmd -global_attention_function ${global_attention_function} "
        cmd="$cmd -self_attn_type ${self_attn_type} "
        cmd="$cmd -generator_function ${generator_function}"
            
        echo $cmd
        CUDA_VISIBLE_DEVICES=`free-gpu` eval $cmd

        models=`ls ${out}_* 2>/dev/null`
        if [ ! -z "$models" ]
        then
          touch $out # cheat
        fi


        if [ -e qsub.stderr]
        then
          ln -s qsub.stderr $train_log
        fi
    }

task model_selection
     < in=$train_log@train
     < modelprefix=$out@train
     > best_model
     :: model_selection_strategy=@ 
     {
        ls ${modelprefix}_* -1v > model_list
        cat $in | grep "Validation acc" | cut -d ' ' -f6 > acc_list
        cat $in | grep "Validation per" | cut -d ' ' -f6 > ppl_list
        paste model_list acc_list ppl_list > all_info
        if [ "$model_selection_strategy" = "acc" ]
        then
            best=`cat all_info | sort -rn -k2 |head -n 1| sed 's/pt.*/pt/'`
        elif [ "$model_selection_strategy" = "ppl" ] ; then
            best=`cat all_info | sort -n -k3 |head -n 1| sed 's/pt.*/pt/'`
        fi
        ln -s $best $best_model
        
      }

task translate : onmt mosesdecoder
      < src=$out@apply_bpe[side:src]
      < tgt=$out@apply_bpe[side:tgt]
      < model=$best_model@model_selection
      > out
      > bleu_scores
      :: section=(section: train='train' dev='dev' test='test')
      :: python_env=@
      :: .submitter=$submitter
      :: .action_flags=$action_flags
      :: .resource_flags=$gpu_resource_flags

      {
          if [ "$section" = "train" ]
          then
              echo "did nothing on train" >  out
              echo "did nothing on train" >  bleu_scores
              exit 0
          fi
          
          model_name=`basename ${model}`
          
          hostname
          GPU=`free-gpu`
          echo "Using GPU $GPU"
          source activate $python_env
          CUDA_VISIBLE_DEVICES=$GPU python ${onmt}/translate.py \
              -gpu 0 \
              -model ${model} \
              -src ${src} \
              -beam_size 12 \
              -replace_unk \
              -output ${out}.bpe

          sed -r 's/\@\@ //g' < ${out}.bpe > ${out}.true
          sed -r 's/\@\@ //g' < ${tgt} > ref.true

          # detruecase
          ${mosesdecoder}/scripts/recaser/detruecase.perl < ${out}.true > ${out}.tok
          ${mosesdecoder}/scripts/recaser/detruecase.perl < ref.true > ref.tok

          # detokenize
          ${mosesdecoder}/scripts/tokenizer/detokenizer.perl < ${out}.tok > ${out}
          ${mosesdecoder}/scripts/tokenizer/detokenizer.perl < ref.tok > ref.detok

          # get BLEU
          blue_detail=`/home/pkoehn/moses/scripts/generic/multi-bleu-detok.perl ref.detok < ${out}`
          BLEU=`echo ${blue_detail}| cut -f 3 -d ' ' | cut -f 1 -d ','`
          echo `date`"$section ${model_name}: ${blue_detail}" >> $bleu_scores
        echo "BLEU = $BLEU"
      }

#task translate_dev calls translate 
#    < src=$out@apply_bpe[section:dev,side:src]
#    < tgt=$out@apply_bpe[section:dev,side:tgt]
#    < model=$best_model@model_selection
#    > out
#    > bleu_scores
#    :: python_env=@
#    :: .submitter=$submitter
#    :: .action_flags=action_flags
#    :: .resource_flags=gpu_resource_flags


#task translate_test calls translate 
#    < src=$out@apply_bpe[section:test,side:src]
#    < tgt=$out@apply_bpe[section:test,side:tgt]
#    < model=$best_model@model_selection
#    > out
#    > bleu_scores
#    :: python_env=@
#    :: .submitter=$submitter
#    :: .action_flags=action_flags
#    :: .resource_flags=gpu_resource_flags

#task report_scores
#    < in=(DecodeSection: dev=$bleu_scores@translate_dev test=$bleu_scores@translate_test)
#    > out
#    {
#      ln -s $in $out
#    }

plan test {
  reach translate via (section: *) * (side: *) * (NumberOfBPE: 40000) * (RunTasksBeforeBPE: no)
}

submitter sge :: action_flags
              :: COMMANDS
              :: TASK REALIZATION TASK_VARIABLES CONFIGURATION {
  action run {
    wrapper="ducttape_job.sh"
    echo "#$ $resource_flags" >> $wrapper
    echo "#$ $action_flags" >> $wrapper
    echo "#$ -o localhost:$PWD/qsub.stdout" >> $wrapper
    echo "#$ -e localhost:$PWD/qsub.stderr" >> $wrapper
    echo "#$ -V" >> $wrapper
    echo "#$ -S /bin/zsh" >> $wrapper
    echo "#$ -N $CONFIGURATION-$TASK-$REALIZATION" >> $wrapper
    echo "set -e # stop on errors" >> $wrapper

    echo "$TASK_VARIABLES" | grep -v "resource_flags" | grep -v "action_flags" >> $wrapper
    echo "cd $PWD" >> $wrapper
    echo "$COMMANDS" >> $wrapper

    # Use SGE's -sync option to prevent qsub from immediately returning
    qsub -sync y $wrapper
  }
}

versioner git :: repo ref {
  action checkout > dir {
    git clone $repo $dir
  }
  action repo_version > version {
    git ls-remote $repo $ref | cut -f1 > $version
  }
  # Used to confirm version after checkout
  action local_version > version date {
    git rev-parse HEAD > $version
    git log -1 | awk '/^Date/{$1=""; print}' > $date
  }
}


# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

