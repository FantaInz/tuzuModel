#!/bin/bash

new_files=()

for notebook in *.ipynb; do
    script_name="${notebook%.ipynb}.py"
    
    if [[ ! -f "$script_name" ]]; then 
        echo "Converting $notebook to Python script..."
        jupyter nbconvert --to script "$notebook"
        new_files+=("$script_name")
    fi
done

for file in "${new_files[@]}"; do
    new_name=$(echo "$file" | sed 's/ /_/g' | sed 's/\./_/1' | sed 's/__/_/g')
    
    if [[ "$file" != "$new_name" ]]; then
        mv "$file" "$new_name"
        file="$new_name"
    fi
    converted_files+=("$file")
done

sed -i 's/plt.show()/plt.close("all")/g' *.py

for script in {1..9}_*.py; do
    if [[ -f "$script" ]]; then
        echo "Running $script..."
        python3 "$script"
        
        if [ $? -eq 0 ]; then
            echo "Successfully executed $script."
        else
            echo "Error occurred while executing $script."
        fi
    fi
done

if [ "${RUN_SAVE_TO_DB}" == "true" ]; then
    echo "RUN_SAVE_TO_DB is true. Running 10_save_to_database.py..."
    python3 10_save_to_database.py

    if [ $? -eq 0 ]; then
        echo "Successfully executed 10_save_to_database.py."
        new_files+=("10_save_to_database.py")
    else
        echo "Error occurred while executing 10_save_to_database.py."
    fi
else
    echo "RUN_SAVE_TO_DB is not true. Skipping 10_save_to_database.py..."
fi


if [[ ${#converted_files[@]} -gt 0 ]]; then
    echo "Cleaning up converted Python files..."
    rm -f "${converted_files[@]}"
fi

echo "All scripts executed and cleaned up."