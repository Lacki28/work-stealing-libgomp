#file: Makefile
#author: Anna Lackinger 11776842
#date: 13.10.2020


CC=gcc
DEFS = -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
CFLAGS = -Wall -g -std=c99 -pedantic $(DEFS) -fopenmp

OBJECTS = libgomp-test.o queue.o

.PHONY: all clean
all: libgomp-test

libgomp-test: $(OBJECTS)
	$(CC) -o $@ -fopenmp $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@  $<

libgomp-test.o: libgomp-test.c queue.h
queue.o: queue.c queue.h

clean:
	rm -rf *.o libgomp-test
