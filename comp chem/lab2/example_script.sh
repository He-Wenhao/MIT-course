#!/bin/bash

#Set the initial variables 
a=3
b=4

for i in {1..20} #start the for loop
do
#Calculate the sum of a and b. "bc" is a programe that does arithmetic in bash
added_together=$(echo $a+$b|bc)

#Print the output statement
echo $a+$b is $added_together

#Update the values of the variables for the next loop
a=$b
b=$added_together
done

