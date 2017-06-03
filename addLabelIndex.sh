#!/bin/bash

path="src/main/resources/"

OutFileName="src/main/resources/labelIndexAdded"
extension=".csv"
i=0

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do 
				awk -F"," 'BEGIN { OFS = "," } {$4='"$i"'; print}' "$filename" > "$OutFileName$i$extension" 
				i=$(( $i + 1 ))
			done;
		
		fi
done;


