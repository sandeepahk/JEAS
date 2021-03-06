#   Joint Entity-Apect-Sentiment (JEAS) Model  #

----------


- Written by Sandeepa Kannangara, University of New South Wales, s.kannangara@unsw.edu.au, part of code is from http://gibbslda.sourceforge.net/.

- This is a Java implementation of the joint entity-aspect-sentiment (JEAS) model for sentiment, entity and aspect classification and extracting sentiment, entity and aspect topics from text copara. Please refer following publications related to this:
http://unsworks.unsw.edu.au/fapi/datastream/unsworks:62271/SOURCE02?view=true
https://link.springer.com/chapter/10.1007/978-3-030-29894-4_46

- Two variation of JEAS based on the order of latent variable generation called Joint Entity-Aspect-Sentiment-Reverse (JEAS-Reverse) and Joint Entity-Aspect-Sentiment-Hierarchy (JEAS-Hierarchy) also included here

## Parameter Estimation ##

Suppose that the current working directory is the home directory of JESI.

    $ -est [-alpha_e <double>] [-alpha_s <double>] [-alpha_t <double>] [-beta_e <double>] [-beta_s <double>] [-beta_t <double>] [-etopics <int>] [-stopics <int>] [-etopict <int>] [-niters <int>] [-savestep <int>] [-twords <int>] –dir <string> -dfile <string>

in which (parameters in [ ] are optional):



- `-est`: Estimate the JESI model from scratch
- `-alpha_e <double>`: The value of alpha_e, a hyper-parameter of JESI
- `-alpha_s <double>`: The value of alpha_s, a hyper-parameter of JESI
- `-alpha_t <double>`: The value of alpha_t, a hyper-parameter of JESI
- `-beta_e <double>`: The value of beta_e, a hyper-parameter of JESI
- `-beta_s <double>`: The value of beta_s, a hyper-parameter of JESI
- `-beta_t <double>`: The value of beta_t, a hyper-parameter of JESI
- `-etopics <int>`: The number of entities
- `-stopics <int>`: The number of sentiment
- `-ttopics <int>`: The number of aspect
- `-niters <int>`: The number of Gibbs sampling iterations
- `-savestep <int`>: The step (counted by the number of Gibbs sampling iterations) at which the JESI model is saved to hard disk
- `-twords <int>`: The number of most likely words for each topic. The default value is zero. If you set this parameter a value larger than zero, e.g., 20, JESI will print out the list of top 20 most likely words per each topic each time it save the model to hard disk according to the parameter savestep above.
- `-dir <string>`: The input training data directory
- `-dfile <string>`: The input training data file

## Input Data Format ##
The input data should have the format as follows:
    
    [M]
    [document1]
    [document2]
    ...
    [documentM]

in which the first line is the total number for documents `[M]`. Each line after that is one document. `[documenti]` is the ith document of the dataset that consists of a list of `Ni` words/terms.

    [documenti] = [wordi1] [wordi2] ... [wordiNi]

in which all `[wordij] (i=1..M, j=1..Ni)` are text strings and they are separated by the blank character.

## Entity and Aspect Seed Word Lists ##

Need to modify the code to set the source of entity and issue seed words in `loadData()` method of the `Model` class.
