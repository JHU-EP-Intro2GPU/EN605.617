# GPU Final Project - Bill Burgett

## Objective:
Use CUDA to brute-force the plaintext value of a hashed password.

## Background:
Hashing is a basic means of encryption and data integrity checks. Today it is most commonly used to store passwords and to validate that files have not been modified.
A hash is a complex one-way function that is applied to an input to generate a unique output. When storing a user's password, most applications choose to store the hashed value of a user's password in their database - that way if someone were to gain access to their database and search for usernames and passwords to try and break into the application, they could not get the value of the password as it appears as a garbled up mess.
Hashing is more a means of obfuscating the data,

## Scope:
This project will serve as a means of demonstrating the power of GPU programming for executing SIMD operations. To show this my code will take an input string of an MD5 hash and iterate through all valid input strings and compare the hashes of the two until a match is found and return it to the user. Initially the password max length and the character set of valid characters will be known, as this will serve as a proof of concept. Timing information and comparisons will be captured.

## Process:
1. Generate an input hash value
2. Load the hash value
3. Dynamically determine GPU specs to maximize block/warp size
4. Cycle through all available inputs to find one with a matching hash value and return the plaintext to the user

### References
Overview of MD5 hash w/ graphics - https://en.wikipedia.org/wiki/MD5  
MD5.h header file - http://www.efgh.com/software/md5.txt
Discussion on hashing algorithms in use today - https://en.wikipedia.org/wiki/Secure_Hash_Algorithms

### Additional Notes
"Salting" the passwords before they are hashed is much more commonplace and secure nowadays. Here are some useful articles and blogs if you want to learn more about them.
https://en.wikipedia.org/wiki/Salt_(cryptography)
https://stackoverflow.com/questions/420843/how-does-password-salt-help-against-a-rainbow-table-attack
https://security.stackexchange.com/questions/100898/why-store-a-salt-along-side-the-hashed-password
