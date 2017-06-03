#!/bin/bash

path="src/main/resources/"

OutFileName="src/main/resources/concat"
extension=".csv"

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do 
				if [ "$filename"  != "$OutFileName" ] ; 
				then 
					tail -n +2  $filename >>  $OutFileName$j$extension
				fi
			done;
			
		fi
done;

sed -i 1i"x,y,z" "$OutFileName$j$extension" # Add the first line
