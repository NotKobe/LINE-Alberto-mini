This is the prof Node mapping to keep track of. 

The **left** side is the node label, followed by the professor name and then the node's he/she is connected to on the **right**

0 Diochnos 1 2 3 6

1 Mansoor 0 2 3

2 McGovern 0 1 3 6

3 Pan 0 1 2 4 5

4 Schroder 3 5

5 Nelson 3 4

6 Stewart (Dr. Wayne) 0 2 7 8

7 Alexander 6 8

8 Muller 6 7


todo for graphing:
for first, second, all (parameter for LINE):
    create embeddings of size=2, 
    write to file & create plots w/ labels=names, 
    then run using TSNE (the embeddings) & plot new embeddings of size = 2 
        then replot and compare with no TSNE 

create embeddings of size = 3 initialy with LINE
repeat for-loop ^ with embedding size = 3 in 3D