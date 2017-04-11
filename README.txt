To compile the program just type "make" and executable called "sssp" will be made.

No change have been made to the initial format for running the program. 

you can run the program in the same format mentioned in the assignment pdf, like this: -

Usage: Required command line arguments (in order):
Input file: E.g., –-input in.txt
	Block size: E.g., –-bsize 512
		Block count: E.g., –-bcount 4
			Output path: E.g., –-output output.txt
				Processing method: E.g., --method bmf (bellman-ford), tpe (to-process-edge), opt (one or two optimizations)
					Shared memory usage:--usesmem yes, or no
						Sync method: –sync incore, or outcore

						
Contains graphs no included in the tgz and we did not do opt as it was optional for Undergraduates.
Source and Dest sorted graph files were made using custom sorter.

These are some of the command line arguments used to test the program :
 
	./sssp --input p2p-Gnutella04.txt --bsize 256 --bcount 8 --output outputI.txt --method tpe --usesmem no --sync incore
	./sssp --input p2p-Gnutella04.txt --bsize 384 --bcount 5 --output outputI.txt --method tpe --usesmem no --sync incore
	./sssp --input p2p-Gnutella04.txt --bsize 512 --bcount 4 --output outputI.txt --method tpe --usesmem no --sync incore
	./sssp --input p2p-Gnutella04.txt --bsize 768 --bcount 2 --output outputI.txt --method tpe --usesmem no --sync incore
	./sssp --input p2p-Gnutella04.txt --bsize 1024 --bcount 2 --output outputI.txt --method tpe --usesmem no --sync incore
	./sssp --input p2p-Gnutella04.txt --bsize 256 --bcount 8 --output outputO.txt --method tpe --usesmem no --sync outcore
	./sssp --input p2p-Gnutella04.txt --bsize 384 --bcount 5 --output outputO.txt --method tpe --usesmem no --sync outcore
	./sssp --input p2p-Gnutella04.txt --bsize 512 --bcount 4 --output outputO.txt --method tpe --usesmem no --sync outcore
	./sssp --input p2p-Gnutella04.txt --bsize 768 --bcount 2 --output outputO.txt --method tpe --usesmem no --sync outcore
	./sssp --input p2p-Gnutella04.txt --bsize 1024 --bcount 2 --output outputO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Source.txt --bsize 256 --bcount 8 --output outputSourceI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Source.txt --bsize 384 --bcount 5 --output outputSourceI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Source.txt --bsize 512 --bcount 4 --output outputSourceI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Source.txt --bsize 768 --bcount 2 --output outputSourceI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Source.txt --bsize 1024 --bcount 2 --output outputSourceI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Source.txt --bsize 256 --bcount 8 --output outputSourceO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Source.txt --bsize 384 --bcount 5 --output outputSourceO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Source.txt --bsize 512 --bcount 4 --output outputSourceO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Source.txt --bsize 768 --bcount 2 --output outputSourceO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Source.txt --bsize 1024 --bcount 2 --output outputSourceO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 256 --bcount 8 --output outputDestI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Dest.txt --bsize 384 --bcount 5 --output outputDestI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Dest.txt --bsize 512 --bcount 4 --output outputDestI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Dest.txt --bsize 768 --bcount 2 --output outputDestI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Dest.txt --bsize 1024 --bcount 2 --output outputDestI.txt --method tpe --usesmem no --sync incore
	./sssp --input Nutella04Dest.txt --bsize 256 --bcount 8 --output outputDestO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 384 --bcount 5 --output outputDestO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 512 --bcount 4 --output outputDestO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 768 --bcount 2 --output outputDestO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 1024 --bcount 2 --output outputDestO.txt --method tpe --usesmem no --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 256 --bcount 8 --output outputDestS.txt --method tpe --usesmem yes --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 384 --bcount 5 --output outputDestS.txt --method tpe --usesmem yes --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 512 --bcount 4 --output outputDestS.txt --method tpe --usesmem yes --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 768 --bcount 2 --output outputDestS.txt --method tpe --usesmem yes --sync outcore
	./sssp --input Nutella04Dest.txt --bsize 1024 --bcount 2 --output outputDestS.txt --method tpe --usesmem yes --sync outcore
	
Should pring out something like this on the console and will make a output file....

The total computation kernel time on GPU 0 is 0.335 milli-seconds
The total filtering kernel time on GPU 0 is 1.037 milli-seconds
Done.
