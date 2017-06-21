#!/bin/bash

path="src/main/resources/"

OutFileName="src/main/resources/concat"
extension=".csv"
j=0

for file in "$path"*; do
		if [ -d $file ]
		then
			
			echo "$file is a directory"
			
			for filename in "$file"/*.csv; do 
				if [ "$filename"  != "$OutFileName" ] ;      # Avoid recursion 
				then 
					tail -n +2  $filename >>  $OutFileName$j$extension # Append from the 2nd line each file
				fi
			done;
			
			j=$(( $j + 1 ))                        # Increase the counter
		
		fi
done;

j=$(( $j - 1 )) 

until [  $j -lt 0 ]; do
	sed 's/^ *//' "$OutFileName$j$extension" | cut -d "," -f2- > "$path"final"$j$extension" # Delete the first column
	sed -i 1i"x,y,z" "$path"final"$j$extension"
	j=$(( $j - 1 ))
done




