"""
Process the raw data into: 

1. one file that contains all the sentence
2. one file contains the correponded relation ship
[{origin:(record_num, line), 
 line: 20, c1:("concept 1", 0, 1), 
 r:"TrCP", c2:("concept 2", 7, 8)
},...,]
3. one file contains the segment of the new sentence:



....

Eventally, get a tensor:
each word is a nw embedding

"""
