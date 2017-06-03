#!/bin/bash

path="src/main/resources/"

OutFileName="src/main/resources/concat.csv"                       # Fix the output name
i=0                                       # Reset a counter

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do 
				if [ "$filename"  != "$OutFileName" ] ;      # Avoid recursion 
				then 
					if [[ $i -eq 0 ]] ; then 
						head -1  $filename >   $OutFileName # Copy header if it is the first file
					fi
					tail -n +2  $filename >>  $OutFileName # Append from the 2nd line each file
					i=$(( $i + 1 ))                        # Increase the counter
				fi
			done;
			
		fi
done;

sed 's/^ *//' "$OutFileName" | cut -d "," -f2- > "$path"final.csv # Delete the first column
