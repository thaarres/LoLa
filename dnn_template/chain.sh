for NC in 10
do 
for i in 1 2 3 4 5
do python SimpleTrainAWS.py $NC $i > out$NC_$i.txt; 
done
done

#python SimpleTrainAWS.py > out1.txt
#python SimpleTrainAWS.py > out2.txt
#python SimpleTrainAWS.py > out3.txt
#python SimpleTrainAWS.py > out4.txt
#python SimpleTrainAWS.py > out5.txt
