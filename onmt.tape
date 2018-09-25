package fairseq
    :: .versioner=git .repo="https://github.com/shuoyangd/fairseq" .ref=HEAD
    :: realref="refs/remotes/origin/master" {

  git checkout -b working $realref
  set +u
  source activate mxnet_v1.2.1
  set -u
  python setup.py build develop
}

package subword_nmt :: .versioner=git .repo="https://github.com/rsennrich/subword-nmt" .ref=HEAD {

}

package mosesdecoder :: .versioner=git .repo="https://github.com/moses-smt/mosesdecoder" .ref=HEAD {

}

package stanford_seg :: .versioner=disk .path="/home/shuoyangd/stanford-seg" {

}

# there seems to be a bug in branching, so this is necessary
task aggregate
    < in=(DataSection: train=$train_data devtest=(DevtestDataSection: dev=$dev_data test=$test_data))
    > out {

  ln -s $in $out
}

task train_sample
    < src_in=$out@aggregate[DataSection:train,side:src]
    < trg_in=$out@aggregate[DataSection:train,side:trg]
    > src_out
    > trg_out
    :: working_dir=@
    :: train_sample_size=(TrainSampleSize: 100000 250000 500000 1000000 DontSample="") {

  linen=`wc -l $src_in | awk '{ print $1}'`
  if [ -z "$train_sample_size" ] || [[ "$train_sample_size" -ge "$linen" ]] ; then
    tee < $src_in > $src_out
    tee < $trg_in > $trg_out
  else
    ln -s $src_in $PWD/src_in
    ln -s $trg_in $PWD/trg_in
    python $working_dir/random_parallel_sample.py $train_sample_size $PWD/src_in $PWD/trg_in
    mv $PWD/src_in.sample $src_out
    mv $PWD/trg_in.sample $trg_out
  fi
}

task dev_text_from_sgm : mosesdecoder
    < src_in=$out@aggregate[DevtestDataSection:dev,side:src]
    < trg_in=$out@aggregate[DevtestDataSection:dev,side:trg]
    < wrap_template=$out@aggregate[DevtestDataSection:dev,side:src]
    > src_out
    > trg_out
    :: sgm_dev=@ {

  if [ ! -z $sgm_dev ] ; then
    $mosesdecoder/scripts/ems/support/input-from-sgm.perl < $src_in > $src_out
  else
    cp $src_in $src_out
  fi

  if [ ! -z $sgm_dev ] ; then
    $mosesdecoder/scripts/ems/support/reference-from-sgm.perl $trg_in $wrap_template $trg_out
  else
    cp $trg_in $trg_out
  fi

  # if there are multi-reference, just use the first one
  if [ -f ${trg_out}.ref0 ] ; then
    mv ${trg_out}.ref0 $trg_out
  fi

  if [ -f ${trg_out}0 ] ; then
    mv ${trg_out}0 $trg_out
  fi
}

task test_text_from_sgm : mosesdecoder
    < src_in=$out@aggregate[DevtestDataSection:test,side:src]
    < trg_in=$out@aggregate[DevtestDataSection:test,side:trg]
    < wrap_template=$out@aggregate[DevtestDataSection:test,side:src]
    > src_out
    > trg_out
    :: sgm_test=@ {

  if [ ! -z $sgm_test ] ; then
    $mosesdecoder/scripts/ems/support/input-from-sgm.perl < $src_in > $src_out
  else
    cp $src_in $src_out
  fi

  if [ ! -z $sgm_test ] ; then
    $mosesdecoder/scripts/ems/support/reference-from-sgm.perl $trg_in $wrap_template $trg_out
  else
    cp $trg_in $trg_out
  fi

  refn=`ls $trg_out* | wc -l`
  if [ $refn -gt 1 ] ; then
    touch $trg_out # cheat
  fi
}

func tokenize
    < in
    > out
    :: working_dir
    :: Lang {

  # segmentation for chinese
  if [ $Lang == "zh" ] || [ $Lang == "chn" ] || [ $Lang == "cn" ] ; then

    mkdir -p $PWD/tmp
    $stanford_seg/segmentstd.sh $stanford_seg/segment.sh $PWD/tmp ctb UTF-8 0 < $in | $mosesdecoder/scripts/tokenizer/escape-special-chars.perl | $working_dir/chinese-punctuations-utf8.perl > $out
    rm -r $PWD/tmp

  else
    $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $Lang < $in | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $Lang > $out
  fi
}

task src_tokenize calls tokenize : stanford_seg mosesdecoder
    < in=(DevtestDataSection: dev=$src_out@dev_text_from_sgm test=$src_out@test_text_from_sgm)
    > out
    :: working_dir=@
    :: Lang=$SRC
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags

task trg_tokenize calls tokenize : stanford_seg mosesdecoder
    < in=(DevtestDataSection: dev=$trg_out@dev_text_from_sgm test=$trg_out@test_text_from_sgm)
    > out
    :: working_dir=@
    :: Lang=$TRG
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags

task train_truecaser : mosesdecoder
    < in=(side: src=$src_out@train_sample trg=$trg_out@train_sample)
    > out
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  for in_file in $in ; do
    cat $in_file >> $PWD/tmp
  done

  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $PWD/tmp -model $out

  rm $PWD/tmp
}

task truecase : mosesdecoder
    < in=(DataSection: train=(side: src=$src_out@train_sample trg=$trg_out@train_sample) devtest=(DevtestDataSection: dev=(side: src=$out@src_tokenize[DevtestDataSection:dev] trg=$out@trg_tokenize[DevtestDataSection:dev]) test=(side: src=$out@src_tokenize[DevtestDataSection:test] trg=$out@trg_tokenize[DevtestDataSection:test])))
    < model=(side: src=$src_truecaser trg=$trg_truecaser)
    > out
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  $mosesdecoder/scripts/recaser/truecase.perl -model $model < $in > $out
}

task train_bpe : subword_nmt
      < src_in=$bpe_input[DataSection:train,side:src]
      < trg_in=$bpe_input[DataSection:train,side:trg]
      > out
      :: bpe_operations=@
      :: SRC=@
      :: TRG=@
      :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  cat $src_in $trg_in | $subword_nmt/learn_bpe.py -s $bpe_operations > $out
}

task apply_bpe : subword_nmt
    # < in=(DataSection:train=$out@merge devtest=(DevtestDataSection: dev=$out@merge_dev test=$out@truecase[DataSection:devtest,DevtestDataSection:test]))
    < in=$bpe_input
    < model=$out@train_bpe
    > out {

  $subword_nmt/apply_bpe.py -c $model < $in > $out
}

task binarize_data : fairseq
    < train_src_in=$out@apply_bpe[DataSection:train,side:src]
    < train_trg_in=$out@apply_bpe[DataSection:train,side:trg]
    < dev_src_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:dev,side:src]
    < dev_trg_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:dev,side:trg]
    < test_src_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:src]
    < test_trg_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:trg]
    > out
    :: conda_env=@
    # :: train_max_shard_size=@
    :: SRC=@
    :: TRG=@
    :: .submitter=$submitter
    :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
    :: .action_flags="-m ae -M xutai_ma@jhu.edu" {


  source activate $conda_env

  ln -s $train_src_in train.$SRC
  ln -s $train_trg_in train.$TRG
  ln -s $dev_src_in dev.$SRC
  ln -s $dev_trg_in dev.$TRG
  ln -s $test_src_in test.$SRC
  ln -s $test_trg_in test.$TRG

  python $fairseq/preprocess.py --source-lang $SRC --target-lang $TRG \
    --trainpref train --validpref dev --testpref test \
    --destdir $out

  if [ -f ${out}.train.1.pt ] ; then
    touch $out
  fi
}

task train : fairseq
    < in=$out@binarize_data
    > out
    :: conda_env=@
    :: train_batch_size=@
    :: train_optim=@
    :: train_dropout=@
    :: train_adam_beta1=@
    :: train_adam_beta2=@
    :: train_lr=@
    :: train_lr_min=@
    :: train_lr_shrink=@
    :: train_lr_scheduler=@
    :: train_warmup_init_lr=@
    :: train_warmup_updates=@
    :: train_criterion=@
    :: train_label_smoothing=@
    :: train_clip_norm=@
    :: train_max_tokens=@
    # :: train_encoder_type=@
    :: train_arch=@
    :: .submitter=$submitter
    :: .resource_flags="-l 'hostname=b1[23456789]|c*,gpu=1,mem_free=4g,ram_free=4g'"
    :: .action_flags="-m bae -M xutai_ma@jhu.edu -q g.q" {

  sleep `od -An -N1 -i /dev/random` # avoid GPU clash

  source activate $conda_env
  export device=`free-gpu`
  echo "gpu$device at "`hostname`
  if [ -z $device ] ; then
    echo "no device! grid cheaaaaaaaaaatin!"
    exit;
  fi

  cmd="python $fairseq/train.py $in --save-dir $out"
  cmd=$cmd" --lr $train_lr --clip-norm $train_clip_norm --dropout $train_dropout --max-tokens $train_max_tokens --arch $train_arch"

  if [ ! -z $train_optim ] ; then
    cmd=$cmd" --optimizer $train_optim"
  fi

  if [ ! -z $train_lr_min ] ; then
    cmd=$cmd" --min-lr $train_lr_min"
  fi

  if [ ! -z $train_lr_shrink ] ; then
    cmd=$cmd" --lr-shrink $train_lr_shrink"
  fi

  if [ ! -z $train_lr_scheduler ] ; then
    cmd=$cmd" --lr-scheduler $train_lr_scheduler"
  fi

  if [ ! -z $train_warmup_init_lr ] ; then
    cmd=$cmd" --warmup-init-lr $train_warmup_init_lr"
  fi

  if [ ! -z $train_warmup_updates ] ; then
    cmd=$cmd" --warmup-updates $train_warmup_updates"
  fi

  if [ ! -z $train_criterion ] ; then
    cmd=$cmd" --criterion $train_criterion"
  fi

  if [ ! -z $train_label_smoothing ] ; then
    cmd=$cmd" --label-smoothing $train_label_smoothing"
  fi

  if [ ! -z $train_max_tokens ] ; then
    cmd=$cmd" --max-tokens $train_max_tokens"
  fi

  if [ ! -z $train_adam_betas ] ; then
    cmd=$cmd" --adam-betas ($train_adam_beta1, $train_adam_beta2)"
  fi

  if [ ! -z $train_share_all_embeddings ] ; then
    cmd=$cmd" --share-all-embeddings"
  fi

  echo $cmd
  CUDA_VISIBLE_DEVICES=`free-gpu` eval $cmd

  # models=`ls ${out}_* 2>/dev/null`
  # if [ ! -z "$models" ] ; then
  #   touch $out # cheat
  # fi
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
task decode : fairseq
    # < src_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:src]
    # < trg_in=$out@apply_bpe[DataSection:devtest,DevtestDataSection:test,side:trg]
    < in=$out@binarize_data
    < model=$out@train
    > out
    :: test_max_sent_length=@
    :: test_beam_size=@
    :: test_batch_size=@
    # :: test_replace_unk=@
    :: test_remove_bpe=@
    :: .submitter=$submitter
    :: .action_flags="-m ae -M xutai_ma@jhu.edu -q g.q"
    :: .resource_flags="-l 'hostname=b1[23456789]|c*,gpu=1,mem_free=4g,ram_free=4g'"
    :: conda_env=@ {

  # . /export/home/sding/anaconda3/etc/profile.d/conda.sh  # start conda

  conda activate $conda_env
  refn=`ls $trg_in* | wc -l`

  # if [ $refn -eq 1 ]; then
  #   cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -tgt $trg_in -replace_unk -verbose -output $out -batch_size 1 -beam_size 12"
    # cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -verbose -output $out -batch_size $test_batch_size -beam_size $test_beam_size -max_sent_length $test_max_sent_length"
  # else
  #   cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -tgt $trg_in0 -replace_unk -verbose -output $out -batch_size 1 -beam_size 12"
    # cmd="python $fairseq/translate.py -gpu $device -model $model -src $src_in -verbose -output $out -batch_size $test_batch_size -beam_size $test_beam_size -max_sent_length $test_max_sent_length"
  # fi

  # if [ ! -z $test_replace_unk ]; then
  #   cmd=$cmd" -replace_unk"
  # fi

  cmd="python $fairseq/generate.py $in --path $model/checkpoint_best.pt --batch-size $test_batch_size --beam $test_beam_size"

  if [ ! -z $test_remove_bpe ] ; then
    cmd=$cmd" --remove-bpe"
  fi

  echo $cmd
  CUDA_VISIBLE_DEVICES=`free-gpu` $cmd > decode.log
  cat decode.log | grep ^H | cut -f3- > $out
}

task debpe
    < in=$out@decode
    > out
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  cat $in | sed -r 's/\@\@ //g' > $out
}

task detruecase : mosesdecoder
    < in=$out@debpe
    > out
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  $mosesdecoder/scripts/recaser/detruecase.perl < $in > $out
}

task detokenize : mosesdecoder
    < in=$out@detruecase
    > out
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  $mosesdecoder/scripts/tokenizer/detokenizer.perl < $in > $out
}

task wrap_xml : mosesdecoder
    < in=$out@detokenize
    < wrap_template=$out@aggregate[DevtestDataSection:test,side:src]
    > out
    :: trg_lang=@
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  $mosesdecoder/scripts/ems/support/wrap-xml.perl $trg_lang $wrap_template < $in > $out
}

task nist_bleu : mosesdecoder
    < in=$out@wrap_xml
    < wrap_template=$out@aggregate[DevtestDataSection:test,side:src]
    < ref=$out@aggregate[DataSection:devtest,DevtestDataSection:test,side:trg]
    > bleu
    > bleu_c
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {


  $mosesdecoder/scripts/generic/mteval-v13a.pl -s $wrap_template -r $ref -t $in > $bleu
  $mosesdecoder/scripts/generic/mteval-v13a.pl -c -s $wrap_template -r $ref -t $in > $bleu_c
}

task multi_bleu : mosesdecoder
    < in=$out@detokenize
    < ref=$tokenized_trg
    > bleu
    > bleu_c
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  refn=`ls $ref* | wc -l`
  if [ $refn -gt 1 ] ; then
    mv $ref ${ref}.tmp # get rid of dummy single reference
  fi

  $mosesdecoder/scripts/generic/multi-bleu-detok.perl $ref < $in > $bleu_c
  $mosesdecoder/scripts/generic/multi-bleu-detok.perl -lc $ref < $in > $bleu

  if [ $refn -gt 1 ] ; then
    mv ${ref}.tmp $ref
  fi
}

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
  # reach multi_bleu via (UseExistingTruecaser: no) * (TrainSampleSize: DontSample) * (DataSource: de_en) * (Architecture: fconv transformer) * (ClipNorm: 0.1 1 5)
  # reach multi_bleu via (UseExistingTruecaser: no) * (TrainSampleSize: DontSample) * (DataSource: de_en) * (Architecture: fconv_iwslt_de_en) * (ClipNorm: 0.1)
  reach multi_bleu via (UseExistingTruecaser: no) * (TrainSampleSize: DontSample) * (DataSource: so_en) * (Architecture: fconv_iwslt_de_en transformer_iwslt_de_en) * (NumberOfBPE: *) * (ClipNorm: 5) * (DoTokenizeAndTruecase: no) * (DoTokenize: no)
  #reach multi_bleu via (UseExistingTruecaser: no) * (TrainSampleSize: DontSample) * (DataSource: so_en) * (Architecture: transformer_iwslt_de_en) * (ClipNorm: 5) * (DoTokenizeAndTruecase: no) * (DoTokenize: no)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

