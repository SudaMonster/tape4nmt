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
  < in=(Split: train=$train_data dev=$dev_data test=$test_data)
  > out
  :: lang=(side: src=$src tgt=$tgt)
  {
      $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang < $in | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $lang > $out
  }


task train_truecaser : mosesdecoder
    < in=$out@tokenize[Split:train]
    > out
    {
      $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $in -model $out
    }


task truecase : mosesdecoder
    < in=$out@tokenize
    < model=$out@train_truecaser
    > out
    {
    $mosesdecoder/scripts/recaser/truecase.perl -model $model < $in > $out
    }


task train_bpe : subword_nmt
    < in=$train_bpe_input
    > out
    :: bpe_operations=@
    {
      cat $in | $subword_nmt/learn_bpe.py -s $bpe_operations > $out
    }


task apply_bpe : subword_nmt
    < in=$apply_bpe_input
    < model=$out@train_bpe
    > out 
    {
      $subword_nmt/apply_bpe.py -c $model < $in > $out
    }

task clean_corpus : mosesdecoder
    < train_src_in=$out@apply_bpe[Split:train,side:src]
    < train_tgt_in=$out@apply_bpe[Split:train,side:tgt]
    > train_src_out
    > train_tgt_out
    :: src=@
    :: tgt=@
    :: clean_minlen=@
    :: clean_maxlen=@
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
    < dev_src_in=$out@apply_bpe[Split:dev,side:src]
    < dev_tgt_in=$out@apply_bpe[Split:dev,side:tgt]
    < test_src_in=$out@apply_bpe[Split:test,side:src]
    < test_tgt_in=$out@apply_bpe[Split:test,side:tgt]
    > out
    :: python_env=@
    :: src=@
    :: tgt=@
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
    {
        source activate $python_env
      
        cmd="python -u $onmt/train.py 
            -data $in 
            -save_model $out 
            -save_checkpoint_steps ${save_checkpoint_steps} 
            -keep_checkpoint ${keep_checkpoint} 
            -seed ${seed} 
            -param_init ${param_init} 
            -batch_size ${batch_size} 
            -batch_type ${batch_type} 
            -normalization ${normalization} 
            -valid_steps ${valid_steps} 
            -valid_batch_size ${valid_batch_size} 
            -max_generator_batches ${max_generator_batches} 
            -train_steps ${train_steps} 
            -optim ${optim} 
            -max_grad_norm ${max_grad_norm} 
            -dropout ${dropout} 
            -adam_beta1 ${adam_beta1} 
            -adam_beta2 ${adam_beta2} 
            -label_smoothing ${label_smoothing} 
            -learning_rate ${learning_rate} 
            -learning_rate_decay ${learning_rate_decay} 
            -start_decay_steps ${start_decay_steps} 
            -decay_steps ${decay_steps} 
            -report_every ${report_every}
            -src_word_vec_size ${src_word_vec_size} 
            -tgt_word_vec_size ${tgt_word_vec_size} 
            -feat_merge ${feat_merge} 
            -feat_vec_size ${feat_vec_size} 
            -feat_vec_exponent ${feat_vec_exponent} 
            -encoder_type ${encoder_type} 
            -decoder_type ${decoder_type} 
            -enc_layers ${enc_layers} 
            -dec_layers ${dec_layers} 
            -rnn_size ${rnn_size} 
            -rnn_type ${rnn_type} 
            -global_attention ${global_attention}
            -global_attention_function ${global_attention_function} 
            -self_attn_type ${self_attn_type} 
            -generator_function ${generator_function}"
            
        echo $cmd
        CUDA_VISIBLE_DEVICES=`free-gpu` eval $cmd

        models=`ls ${out}_* 2>/dev/null`
        if [ ! -z "$models" ] ; then
          touch $out # cheat
        fi
    }

# task model_selection
#     < in=$out@train
#     > out
#     :: test_model_selection_strategy=@ {
#
#   touch list
#   for dump in `ls ${in}_*`; do
#     echo $dump
#     acc=`basename $dump | grep -Eo "acc_[0-9\.]+" | grep -Eo "[0-9\.]+"`
#     ppl=`basename $dump | grep -Eo "ppl_[0-9\.]+" | grep -Eo "[0-9\.]+"`
#     epc=`basename $dump | grep -Eo "e[0-9]+" | grep -Eo "[0-9]+"`
#     if [ "$test_model_selection_strategy" == "acc" ] ; then
#       echo $acc >> list
#     fi
#     if [ "$test_model_selection_strategy" == "ppl" ] ; then
#       echo $ppl >> list
#     fi
#   done
#
#
#   if [ "$test_model_selection_strategy" == "acc" ] ; then
#     best_crit=`sort -g list | tail -1`
#     echo $best_crit
#     best=`ls ${in}_* | grep acc_${best_crit} | tail -1`
#   fi
#   if [ "$test_model_selection_strategy" == "ppl" ] ; then
#     best_crit=`sort -g list | head -1`
#     echo $best_crit
#     best=`ls ${in}_* | grep ppl_${best_crit} | tail -1`
#   fi
#
#   if [ ! -z $best ] ; then
#     echo "$best is selected as the best model according to strategy: $test_model_selection_strategy"
#     cp $best $out
#   else
#     echo "unsupported strategy $test_model_selection_strategy detected"
#   fi
#
#   rm list
# }

# the target input here is used to compute naïve acc and ppl,
# that's why we need post-bpe target input
#task decode : fairseq
#    # < src_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:src]
#    # < trg_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:trg]
#    < in=$out@binarize_data
#    < model=$out@train
#    > out
#    :: test_max_sent_length=@
#    :: test_beam_size=@
#    :: test_batch_size=@
#    # :: test_replace_unk=@
#    :: test_remove_bpe=@
#    :: .submitter=$submitter
#    :: .action_flags="-m ae -M xutai_ma@jhu.edu -q g.q"
#    :: .resource_flags="-l 'hostname=b1[23456789]|c*,gpu=1,mem_free=4g,ram_free=4g'"
#    :: conda_env=@ {
#
#  # . /export/home/sding/anaconda3/etc/profile.d/conda.sh  # start conda
#
#  conda activate $conda_env
#  refn=`ls $trg_in* | wc -l`
#
#  # if [ $refn -eq 1 ]; then
#  #   cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -tgt $trg_in -replace_unk -verbose -output $out -batch_size 1 -beam_size 12"
#    # cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -verbose -output $out -batch_size $test_batch_size -beam_size $test_beam_size -max_sent_length $test_max_sent_length"
#  # else
#  #   cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -tgt $trg_in0 -replace_unk -verbose -output $out -batch_size 1 -beam_size 12"
#    # cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -verbose -output $out -batch_size $test_batch_size -beam_size $test_beam_size -max_sent_length $test_max_sent_length"
#  # fi
#
#  # if [ ! -z $test_replace_unk ]; then
#  #   cmd=$cmd" -replace_unk"
#  # fi
#
#  cmd="python $fairseq/generate.py $in --path $model/checkpoint_best.pt --batch-size $test_batch_size --beam $test_beam_size"
#
#  if [ ! -z $test_remove_bpe ] ; then
#    cmd=$cmd" --remove-bpe"
#  fi
#
#  echo $cmd
#  CUDA_VISIBLE_DEVICES=`free-gpu` $cmd > decode.log
#  cat decode.log | grep ^H | cut -f3- > $out
#}
#
#task debpe
#    < in=$out@decode
#    > out
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#  cat $in | sed -r 's/\@\@ //g' > $out
#}
#
#task detruecase : mosesdecoder
#    < in=$out@debpe
#    > out
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#  $mosesdecoder/scripts/recaser/detruecase.perl < $in > $out
#}
#
#task detokenize : mosesdecoder
#    < in=$out@detruecase
#    > out
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#  $mosesdecoder/scripts/tokenizer/detokenizer.perl < $in > $out
#}
#
#task wrap_xml : mosesdecoder
#    < in=$out@detokenize
#    < wrap_template=$out@aggregate[DevtestDataSection:test,side:src]
#    > out
#    :: trg_lang=@
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#  $mosesdecoder/scripts/ems/support/wrap-xml.perl $trg_lang $wrap_template < $in > $out
#}
#
#task nist_bleu : mosesdecoder
#    < in=$out@wrap_xml
#    < wrap_template=$out@aggregate[DevtestDataSection:test,side:src]
#    < ref=$out@aggregate[DataSection:devtest,DevtestDataSection:test,side:trg]
#    > bleu
#    > bleu_c
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#
#  $mosesdecoder/scripts/generic/mteval-v13a.pl -s $wrap_template -r $ref -t $in > $bleu
#  $mosesdecoder/scripts/generic/mteval-v13a.pl -c -s $wrap_template -r $ref -t $in > $bleu_c
#}
#
#task multi_bleu : mosesdecoder
#    < in=$out@detokenize
#    < ref=$tokenized_trg
#    > bleu
#    > bleu_c
#    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {
#
#  refn=`ls $ref* | wc -l`
#  if [ $refn -gt 1 ] ; then
#    mv $ref ${ref}.tmp # get rid of dummy single reference
#  fi
#
#  $mosesdecoder/scripts/generic/multi-bleu-detok.perl $ref < $in > $bleu_c
#  $mosesdecoder/scripts/generic/multi-bleu-detok.perl -lc $ref < $in > $bleu
#
#  if [ $refn -gt 1 ] ; then
#    mv ${ref}.tmp $ref
#  fi
#}
#
# COMMANDS: the bash commands from some task
# TASK, REALIZATION, CONFIGURATION: variables passed by ducttape
submitter sge :: action_flags
              :: COMMANDS
              :: TASK REALIZATION TASK_VARIABLES CONFIGURATION {
  action run {
    wrapper="ducttape_job.sh"
    echo "#$ $resource_flags" >> $wrapper
    echo "#$ $action_flags" >> $wrapper
    echo "#$ -j y" >> $wrapper
    echo "#$ -V" >> $wrapper
    echo "#$ -S /bin/zsh" >> $wrapper
    echo "#$ -o localhost:$PWD/job.out" >> $wrapper
    echo "#$ -N $CONFIGURATION-$TASK-$REALIZATION" >> $wrapper

    # Bash flags aren't necessarily passed into the scheduler
    # so we must re-initialize them

    # these two doesn't work well with virtualenv ...
    # echo "set -e # stop on errors" >> $wrapper
    # echo "set -u # stop on undeclared variables" >> $wrapper


    echo "set -o pipefail # stop on pipeline errors" >> $wrapper
    echo "set -x # show each command as it is executed" >> $wrapper
    echo "$TASK_VARIABLES" | grep -v "resource_flags" | grep -v "action_flags" >> $wrapper

    # The current working directory will also be changed by most schedulers
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

plan test {
  reach train via (Split: *) * (side: *) * (NumberOfBPE: 2000) * (RunTasksBeforeBPE: no)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

