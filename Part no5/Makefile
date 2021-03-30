#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for programs running a simple k-means clustering */
#                  algorithm                                                 */
#                                                                            */
#    Author:  Wei-keng Liao                                                  */
#             ECE Department Northwestern University                         */
#             email: wkliao@ece.northwestern.edu                             */
#                                                                            */
#    Copyright (C) 2005, Northwestern University                             */
#    See COPYRIGHT notice in top-level directory.                            */
#                                                                            */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

.KEEP_STATE:

all: seq 


CC             = icc 

INCFLAGS    = -I.
OPTFLAGS    =  -qopenmp -g -traceback -fast -DNDEBUG
LDFLAGS     = $(OPTFLAGS)


CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)


H_FILES     = kmeans.h

COMM_SRC = file_io.c util.c


SEQ_SRC     = seq_main.c   \
              seq_kmeans.c \
	      wtime.c

SEQ_OBJ     = $(SEQ_SRC:%.c=%.o) $(COMM_SRC:%.c=%.o)

$(SEQ_OBJ): $(H_FILES)

seq_main.o: seq_main.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

seq_kmeans.o: seq_kmeans.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

wtime.o: wtime.c
	$(CC) $(CFLAGS) -c $*.c

seq: seq_main
seq_main: $(SEQ_OBJ) $(H_FILES)
	$(CC) $(LDFLAGS) -o $@ $(SEQ_OBJ) $(LIBS)

IMAGE_FILES =   color100.txt   color17695.bin   color17695.nc \
                 edge100.txt    edge17695.bin    edge17695.nc \
              texture100.txt texture17695.bin texture17695.nc

INPUTS = $(IMAGE_FILES:%=Image_data/%)

PACKING_LIST = $(COMM_SRC) $(SEQ_SRC) $(H_FILES) \
               Makefile README COPYRIGHT

dist:
	dist_dir=parallel-kmeans \
	&& rm -rf $$dist_dir $$dist_dir.tar.gz\
	&& mkdir -p $$dist_dir/Image_data \
	&& cp $(PACKING_LIST) $$dist_dir \
	&& cp $(INPUTS) $$dist_dir/Image_data \
	&& tar -cf - $$dist_dir | gzip > $$dist_dir.tar.gz \
	&& rm -rf $$dist_dir

clean:
	rm -rf *.o seq_main \
		core* .make.state              \
		*.cluster_centres *.membership \
		*.cluster_centres.nc *.membership.nc \
		Image_data/*.cluster_centres Image_data/*.membership \
		Image_data/*.cluster_centres.nc Image_data/*.membership.nc

check: all
	# sequential K-means ---------------------------------------------------
	./seq_main -q -b -n 4 -i Image_data/color17695.bin
	./seq_main -q    -n 4 -i Image_data/color100.txt

