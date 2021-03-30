# high-performance-computing
5 problem sets of parallel programming on CPU and GPU. University projects for High Performance Computing Systems (Fall 2016).

## Description


#### Part 1: 
Create Mandelbrot set by dividing into N parts, where each calculates has an individual painting work. Implemented by having each part of the work assigned to a thread that returns to the main thread, which paints the results. 

#### Problem 2:
Control the traffic over a 2-way bridge, so that:
- There are no cars moving both ways
- There are no more than N cars on the bridge at any time
- There can't be any car waiting forever
Implemented with 2 threads in each entrance of the bridge, by using semaphores.

#### Part 3:
A roller coaster fits N passengers and starts only when it's full. Passengers get off when the roller coaster has finished the ride and before the new passengers come. Implemented a synchronization between passengers and roller coaster, where there is one thread for passenger and one for the roller coaster.

## Prerequisites
- CUDA
- C
- Make (sudo apt install make)
- GCC (sudo apt install gcc)

## Authors
- [Panagiotis Nikitakis](https://www.linkedin.com/in/panagiotis-nikitakis/)

## Course website
[ECE415 High Performance Computing Systems](https://www.e-ce.uth.gr/studies/undergraduate/courses/ece415/?lang=en)  
