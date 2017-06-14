#!/bin/bash

path="src/main/resources/"
OutFileName="src/main/resources/reduce"
extension=".csv"
i=0

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do 
				cut -d, -f1,-2,-3 $filename > "$OutFileName$i$extension" 
				i=$(( $i + 1 ))
			done;
		
		fi
done;


