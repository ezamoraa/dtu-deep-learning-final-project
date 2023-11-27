#!/bin/sh

script_dir="/zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression"
output_dir_base="$script_dir/output"

# Experiment configurations
#"<name> <trainingpath> <valpath> <iterations> <bottleneck> <other params>"
#"lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 32 32 --demo"
#"lstm_full $script_dir/../data/train/ $script_dir/../data/val/ 32 32"
  # "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 16 32 --demo"
  # "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 8 32 --demo"
  # "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 4 32 --demo"
  # "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 1 32 --demo"
  #   "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 32 32 --demo --checkpoint /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/output/lstm_demo_i32_b32/weights_step=000200000.tar"

#  "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 8 32 --demo --checkpoint /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/output/lstm_demo_i8_b32/weights_step=000420000.tar"
#   "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 4 32 --demo --checkpoint /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/output/lstm_demo_i4_b32/weights_step=000480000.tar"
#   "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 1 32 --demo --checkpoint /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/output/lstm_demo_i1_b32/weights_step=000480000.tar"


experiments=(
  "lstm_demo $script_dir/../data/train/ $script_dir/../data/val/ 16 32 --demo --checkpoint /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/output/lstm_demo_i16_b32/weights_step=000630000.tar"
)

#--checkpoint
### General options
### –- specify queue --
#queue="hpc"
#queue="gpua100"
queue="gpuv100"
#queue="gpua40"

### -- ask for number of cores (default: 1) --
numb_cores=8

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
walltime="24:00"

# request 15GB of system-memory
mem_request="15GB"

### -- set the email address --
email_address="s230025@dtu.dk"

# -- end of LSF options --

#nvidia-smi

# Run experiments in a loop
for experiment in "${experiments[@]}"; do
  # Specify other variables

  # Parse experiment parameters
  set -- $experiment
  experiment_name=$1
  train_data_dir=$2
  val_data_dir=$3
  num_iter=$4
  bottleneck=$5
  experiment_name="$experiment_name""_i""$num_iter""_b""$bottleneck"
  additional_params=${@:6}

  output_dir="$output_dir_base/$experiment_name"
  mkdir -p $output_dir
  output_file="$output_dir/log_$experiment_name.out"
  error_file="$output_dir/log_$experiment_name.err"

  # Run the command with variables
  echo "Running experiment: $experiment_name"
  bsub -q $queue -J $experiment_name -W $walltime -R "span[hosts=1]" -R "rusage[mem=$mem_request]" -u $email_address -B -N -o $output_file -e $error_file -n $numb_cores -gpu "num=1" <<EOF
module load python3/3.11.4
module load cuda/11.8
module load cudnn/v8.9.1.23-prod-cuda-11.X
module load tensorrt/8.6.1.6-cuda-11.X
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/11.8.0
source /zhome/11/1/193832/resquivel/deepL/dtu-deep-learning-final-project/range_image_compression/.venv/bin/activate
range_image_compression_train --train_data_dir $train_data_dir --val_data_dir $val_data_dir --num_iters $num_iter --bottleneck $bottleneck --train_output_dir $output_dir $additional_params
EOF


done



