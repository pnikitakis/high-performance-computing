#Add any executable you want to be created here.
EXECUTABLES	= lab3

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -arch sm_30 -lineinfo -Xcompiler -Wall -O4 -g 

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm

# make all will create all executables
lab3.c: lab3

# This is the rule to create any executable from the corresponding .c
# file with the same name.
%: %.cu
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -ptxas-options=-v


# make clean will remove all executables, jpg files and the
# output of previous executions.
clean:
	rm -f $(EXECUTABLES)
