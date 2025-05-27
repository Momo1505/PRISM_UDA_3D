#!/bin/bash

# Directories
CONFIG_DIR="configs/mic"
WORK_DIR="work_dirs/local-basic"
OUTPUT_DIR="slurm_scripts"
LAUNCH_SCRIPT="launch_test.sh"
LAUNCH_SCRIPT_RES="launch_test_res.sh"

# Create the output directory
mkdir -p $OUTPUT_DIR

# Create or overwrite the launch script
echo "#!/bin/bash" > $LAUNCH_SCRIPT
echo "" >> $LAUNCH_SCRIPT
echo "# Launch all generated SLURM test scripts" >> $LAUNCH_SCRIPT
echo "" >> $LAUNCH_SCRIPT

echo "#!/bin/bash" > $LAUNCH_SCRIPT_RES
echo "" >> $LAUNCH_SCRIPT_RES
echo "# Launch all generated SLURM test scripts" >> $LAUNCH_SCRIPT_RES
echo "" >> $LAUNCH_SCRIPT_RES

# Iterate through each config file
for CONFIG_FILE in $CONFIG_DIR/*.py; do
    # Extract the "name" field from the config
    NAME=$(grep -Po "(?<=name = ')[^']*" $CONFIG_FILE)
    
    # Find the latest work directory for this name
    LATEST_DIR=$(ls -d $WORK_DIR/*_${NAME}_* 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo "No work directory found for $NAME, skipping..."
        continue
    fi

    # Extract the job name and directory for the SLURM script
    JOB_NAME="test_$NAME"
    SLURM_SCRIPT="$OUTPUT_DIR/${JOB_NAME}.slurm"
    SLURM_SCRIPT_RES="$OUTPUT_DIR/${JOB_NAME}_res.slurm"

    # Extract source and target from the name
    SOURCE_TO_TARGET=${NAME}
    TARGET=${SOURCE_TO_TARGET#*to} # Extract everything after "to"

    # Create the SLURM script
    cat <<EOL > $SLURM_SCRIPT
#! /bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/${JOB_NAME}.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
echo 'START'

TEST_ROOT=$LATEST_DIR
CONFIG_FILE="\${TEST_ROOT}/*\${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="\${TEST_ROOT}/latest.pth"
SHOW_DIR="\${TEST_ROOT}/preds"
echo 'Config File:' \$CONFIG_FILE
echo 'Checkpoint File:' \$CHECKPOINT_FILE
echo 'Predictions Output Directory:' \$SHOW_DIR
python -m tools.test \${CONFIG_FILE} \${CHECKPOINT_FILE} --eval mIoU --show-dir \${SHOW_DIR} --opacity 1

python get_results.py --pred_path \$SHOW_DIR --gt_path ./data/${TARGET}/labels/
EOL

    # Add this script to the launch script
    echo "sbatch $SLURM_SCRIPT" >> $LAUNCH_SCRIPT

    echo "Generated SLURM script for $NAME at $SLURM_SCRIPT"


cat <<EOL > $SLURM_SCRIPT_RES
#! /bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/${JOB_NAME}_res.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
echo 'START'

TEST_ROOT=$LATEST_DIR
CONFIG_FILE="\${TEST_ROOT}/*\${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="\${TEST_ROOT}/latest.pth"
SHOW_DIR="\${TEST_ROOT}/preds"
python get_results.py --pred_path \$SHOW_DIR --gt_path ./data/${TARGET}/labels/
EOL

    # Add this script to the launch script
    echo "sbatch $SLURM_SCRIPT_RES" >> $LAUNCH_SCRIPT_RES

    echo "Generated SLURM script for $NAME at $SLURM_SCRIPT_RES"

done

# Make the launch script executable
chmod +x $LAUNCH_SCRIPT
echo "Launch script created: $LAUNCH_SCRIPT"

chmod +x $LAUNCH_SCRIPT_RES
echo "Launch script res created: $LAUNCH_SCRIPT_RESS"

