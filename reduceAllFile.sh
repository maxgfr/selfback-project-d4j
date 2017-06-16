#!/bin/bash

path="src/main/resources/data"
OutFileName="src/main/resources/data_ultra_light/"
extension=".csv"
j=1

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do
				head -n 50000 $filename > $OutFileName$j$extension
				j=$(( $j + 1 ))
			done;
			
			
		
		fi
done;


