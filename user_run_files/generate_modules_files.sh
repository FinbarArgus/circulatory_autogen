# Check if the required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_model> <output_dir>"
    exit 1
fi

# Assign arguments to variables
INPUT_MODEL=$1
OUTPUT_DIR=$2

source opencor_pythonshell_path.sh
${opencor_pythonshell_path} ../src/scripts/generate_modules_files.py --input-model "$INPUT_MODEL" --output-dir "$OUTPUT_DIR"
