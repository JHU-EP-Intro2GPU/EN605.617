# Instructions

1. Create a CUDA application that utilizes 2 CUDA advanced libraries covered in the last 3 modules (modules 7, 8, and 9). They need to not be from the same module and not be used trivially, meaning that you don't use cuRAND to generate an integer that is never used or create a neural network that doesn't actually compute anything.
2. The more complex and integrated the two libraries usage is the more points that you will earn.
3. The overall assignment must fulfill a real-life use case, even in a non-complete fashion. Think of a solving complex set of linear algebra equations and validating it with random input data or taking images and performing random changes to the RGB values of pixels.
4. Beyond just the code and proof of its execution, create a one-page (font 12 or 14, double spaced, etc. and it can be a little shorter or longer) to discuss what your goals were and any challenges and triumphs that you had.
5. For your assignment submission, you will need to include either a link to the commit/branch for your assignment submission (preferred method), including all code and artifacts, or the zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.

## Available Libraries by Module

- **Module 7**: cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE
- **Module 8**: NPP, nvGraph, Thrust
- **Module 9**: cuDNN, cuTensor

## Assignment Rubric

| Criteria | Proficient | Competent | Adequate | Novice | Pts |
| --- | --- | --- | --- | --- | --- |
| Quality of Use of Advanced Library 1 - You will get a higher score by incorporating the library effectively and efficiently. | 25 to >20.0 pts - Full Marks: Utilize Advanced Library 1 in a manner that shows significant understanding. | 20 to >12.5 pts - Good use of Advanced Library 1: While not the most illustrative or efficient use of the library, it is clear that the user has a complete understanding of the library. | 12.5 to >0.0 pts - Adequate Library Usage: It is clear that the grader has used the library sensibly but it is either clear that the user doesn't understand the library sufficiently or the use of the library does not make complete sense. | 0 pts - No Marks: Did not actually use Advanced Library 1. | 25 pts |
| Quality of Use of Advanced Library 2 - You will get a higher score by incorporating the library effectively and efficiently. | 25 to >20.0 pts - Full Marks: Utilize Advanced Library 2 in a manner that shows significant understanding. | 20 to >12.5 pts - Good use of Advanced Library 2: While not the most illustrative or efficient use of the library, it is clear that the user has a complete understanding of the library. | 12.5 to >0.0 pts - Adequate Library Usage: It is clear that the grader has used the library sensibly but it is either clear that the user doesn't understand the library sufficiently or the use of the library does not make complete sense. | 0 pts - No Marks: Did not actually use Advanced Library 2. | 25 pts |
| Integration of 2 Advanced Libraries into Single Application - You will need to have the use of one library feed into/integrate with the other library. It should be clear that this is happening and the more complex, yet efficient, that solution is the better the score in this criterion will be. | 25 to >12.5 pts - Both Advanced Libraries Integrated Completely: It is clear that the choice of libraries was to complement each other while they are very similar, so no cuDNN and cuTensor. The use of the two libraries is done aiming to be efficient computationally and in the amount of code written. The amount of code is relative to the library choices and steps needed to configure and use them. | 12.5 to >0.0 pts - Adequate Integration of Advanced Libraries: The two libraries are integrated minimally, say only using a random integer generated using cuRAND as the seed for a single calculation vs being used for the initial values for a whole neural network. | | 0 pts - No Integration between the two Advanced libraries. | 25 pts |
| Quality of Code - organization of files/functions, code comments, and constants and lines no longer than 80 characters and 40 lines per function. | 10 pts - Fully Meets Code Quality Requirements | 5 pts - Partially Meets Code Quality Requirements | | 0 pts - Does Not Meet Code Quality Requirements | 10 pts |
| Use of run script and/or makefile - Makefile and/or run.sh script executing all required steps for grading of the assignment. | 5 pts - Makefile or run.sh executes all iterations/variations for grading assignment | | | 0 pts - No Marks | 5 pts |
| Novel, Interesting, or Highly Efficient use of 2 Libraries | 10 pts - Novel, Interesting, or Highly Efficient use of 2 Libraries | 5 pts - Novel, Interesting, or Highly Efficient use of 1 Library | | 0 pts - No Novel, Interesting, or Highly Efficient use of any Libraries | 10 pts |

Total Points: 100
