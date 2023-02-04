# Unobserved Confounders
We apply the paper [Shaking the foundations: delusions in sequence models for interaction and control](https://arxiv.org/abs/2110.10819) on a simple example.

## Language
We define a language `L` over the binary alphabet `Σ={0,1}`. Each word is a symbol drawn from the alphabet. The first word is called `U` the second `Y` and the third `X`.

The following table shows the language `L`:
|U|X|Y|
|---|---|---|
|0|0|0|
|0|0|1|
|0|1|0|
|0|1|1|
|1|0|0|
|1|1|0|
|1|0|1|
|1|1|1|


A sentence is a directed acyclic graph (DAG) with the following structure:
``` Python
U -> X 
U -> Y
X -> Y
```

## Example based on a formal language
We assume that an expert observes the word `U` and generates `X` and `Y` based on this knowledge. The expert chooses to generate the following sentences:
|U|X|Y|
|---|---|---
|0|0|1|
|1|1|1|

Moreover we assume, that we can observe the expert's actions `X` and `Y` but not the word `U`:
|X|Y|
|---|---|
|0|1|
|1|0|

Based on the observations we compute the conditional probability for the word `Y`:
``` Python
P(Y=1|X=0)=1.0
P(Y=1|X=1)=1.0
```
No matter what word `X` is given, we predict that the next word will be `Y=1`. This prediction is only correct if we want to predict the next word given `X` by the expert. But it's not correct if we've generated the word `X`.

We can fix this problem by treating the actions as a causal intervention using the do-operator.
```Python
P(Y=1|do(X)=0))=0.5
P(Y=1|do(X)=1))=0.5
```

## Example based on natural language
A user is asking an expert the following question:
>  Which command can I use to list the files in a directory?

The answer of the expert depends on the user's OS `U`:

|User OS (U)|Word X of answer| Word Y of answer
|---|---|---|
|Linux|`ls` | will work|
|Window|`dir` | will work|

We train a language model on the expert's answers. The language model returns the following probabilities:
``` 
P(will work|ls)=1.0
P(will work|dir)=1.0
``` 

After training we use the language model to answer the question of the user. The user has a Linux OS which we can't observe and asks the following question:
>  Which command can I use to list the files in a directory?

We sample the first word of the answer `dir` with a probability of `0.5`. For the second word `Y` we use the language model which returns `will work` with a probability of `1.0`. The probability is incorrect because we need to know the user's OS to answer the question with certainty.

