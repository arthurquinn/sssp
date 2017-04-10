CC = /usr/local/cuda-7.5/bin/nvcc

sssp: *.cu *.cpp *.c
	$(CC) -std=c++11 parse_graph.cpp utils.c entry_point.cu -O3 -arch=sm_30 -o sssp
#parse_graph.cpp impl1.cu impl2.cu opt.cu 

clean:
	rm -f *.o sssp

test: 
	./sssp --input input.txt --bsize 512 --bcount 192 --output output1.txt --method tpe --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 256 --bcount 8 --output outputI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 384 --bcount 5 --output outputI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 512 --bcount 4 --output outputI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 768 --bcount 2 --output outputI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 1024 --bcount 2 --output outputI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CA-CondMat.txt --bsize 256 --bcount 8 --output outputO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CA-CondMat.txt --bsize 384 --bcount 5 --output outputO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CA-CondMat.txt --bsize 512 --bcount 4 --output outputO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CA-CondMat.txt --bsize 768 --bcount 2 --output outputO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CA-CondMat.txt --bsize 1024 --bcount 2 --output outputO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CASource.txt --bsize 256 --bcount 8 --output outputSourceI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CASource.txt --bsize 384 --bcount 5 --output outputSourceI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CASource.txt --bsize 512 --bcount 4 --output outputSourceI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CASource.txt --bsize 768 --bcount 2 --output outputSourceI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CASource.txt --bsize 1024 --bcount 2 --output outputSourceI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CASource.txt --bsize 256 --bcount 8 --output outputSourceO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CASource.txt --bsize 384 --bcount 5 --output outputSourceO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CASource.txt --bsize 512 --bcount 4 --output outputSourceO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CASource.txt --bsize 768 --bcount 2 --output outputSourceO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CASource.txt --bsize 1024 --bcount 2 --output outputSourceO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 256 --bcount 8 --output outputDestI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CADest.txt --bsize 384 --bcount 5 --output outputDestI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CADest.txt --bsize 512 --bcount 4 --output outputDestI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CADest.txt --bsize 768 --bcount 2 --output outputDestI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CADest.txt --bsize 1024 --bcount 2 --output outputDestI.txt --method bmf --usesmem no --sync incore
	#./sssp --input CADest.txt --bsize 256 --bcount 8 --output outputDestO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 384 --bcount 5 --output outputDestO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 512 --bcount 4 --output outputDestO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 768 --bcount 2 --output outputDestO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 1024 --bcount 2 --output outputDestO.txt --method bmf --usesmem no --sync outcore
	#./sssp --input CADest.txt --bsize 256 --bcount 8 --output outputDestS.txt --method bmf --usesmem yes --sync outcore
	#./sssp --input CADest.txt --bsize 384 --bcount 5 --output outputDestS.txt --method bmf --usesmem yes --sync outcore
	#./sssp --input CADest.txt --bsize 512 --bcount 4 --output outputDestS.txt --method bmf --usesmem yes --sync outcore
	#./sssp --input CADest.txt --bsize 768 --bcount 2 --output outputDestS.txt --method bmf --usesmem yes --sync outcore
	#./sssp --input CADest.txt --bsize 1024 --bcount 2 --output outputDestS.txt --method bmf --usesmem yes --sync outcore