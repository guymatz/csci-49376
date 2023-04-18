# CSCI 49376 - BIG DATA

# Project 2
---------------

## Design of MapReduce Algorithms

## Problem 1

For each drug, compute the number of genes and the number of diseases
associated with the drug. Output results with top 5 number of genes
in descending order

#### Patterns Used

* Filtering
* Summarization
* Join

#### Approach
Using a dataset of graph edges, two RDDs were created using filters, one of compounds and associated diseases, and another of compounds and associated genes.

Both RDDs were mapped to (compound, 1) pairs, then
reduced to get the total count of "Compounds per Disease" and
"Compounds per Gene".  The two RDDs were then converted to 
dataframes and joined on their keys (Compound)

#### Pseudocode
```
class Mapper
    method Map(string t)
        Emit(pair (t, 1))

class Reducer
    method Reduce(pairs [(s_1, c_1),(s_1, c_1)...])
        for all pairs (s,c) \in pairs
            if s_1 == s_2:
                Emit(c_1 + c_2)
```
----
## Problem 2

Compute the number of diseases associated
with 1, 2, 3, …, n drugs. Output results with the top
5 number of diseases in a descending order.
E.g.

```
1 drug -> 2 diseases

2 drugs -> 1 diseases
```

#### Patterns Used

* Filtering
* Summarization

#### Approach
A filter was applied to dataset of graph edges to create
a RDD of compounds and associated diseases

The RDD was then mapped to (compound, 1) pairs, and
reduced to get the total count of "Compounds per Disease".
The data was then "grouped by" the count and sorted.

#### Pseudocode
```
class Mapper
    method Map(string t)
        Emit(pair (t, 1))

class Reducer
    method Reduce(pairs [(s_1, c_1),(s_1, c_1)...])
        for all pairs (s,c) \in pairs
            if s_1 == s_2:
                Emit(c_1 + c_2)
```
----
## Problem 3

Get the name of drugs that have the top 5 number of genes.  Output the results.

```
MagicPill1 -> 2
MagicPill2 -> 1
MagicPill3 -> 0
```

#### Patterns Used

* Filtering
* Summarization
* Join

#### Approach
A filter was applied to dataset of graph nodes to create
a RDD of Compound Names / ID.

A previously created RDD of Compounds/Genes was mapped to (compound, 1) pairs, then reduced to create an RDD of Compounds by 
Gene Count.

The Compound/Gene Count RDD was then joined to the RDD of
Compounds Names / IDs.

#### Pseudocode
```
class Mapper
    method Map(string t)
        Emit(pair (t, 1))

class Reducer
    method Reduce(pairs [(s_1, c_1),(s_1, c_1)...])
        for all pairs (s,c) \in pairs
            if s_1 == s_2:
                Emit(c_1 + c_2)
```


## RESULTS

### Q1.

For each drug, compute the number of genes and the number of diseases
associated with the drug. Output results with top 5 number of genes
in descending order

```
+-----------------+----------+-------------+
|         compound|gene_count|disease_count|
+-----------------+----------+-------------+
|Compound::DB08865|       585|            1|
|Compound::DB01254|       564|            1|
|Compound::DB00997|       532|           17|
|Compound::DB00570|       523|            7|
|Compound::DB00390|       522|            2|
+-----------------+----------+-------------+
```


### Q2:
Compute the number of diseases associated
with 1, 2, 3, …, n drugs. Output results with the top
5 number of diseases in a descending order.
E.g.
1 drug -> 2 diseases
2 drugs -> 1 diseases

```
+----------+-------------+
|drug_count|disease_count|
+----------+-------------+
|         1|          331|
|         2|          107|
|         3|           42|
|         4|           30|
|         5|           10|
+----------+-------------+
```

### Q3:
Get the name of drugs that have the top 5 number of genes.  Output the results.
E.g.:
MagicPill1 -> 2
MagicPill2 -> 1
MagicPill3 -> 0

```
+-----------+----------+
|       name|gene_count|
+-----------+----------+
| Crizotinib|       585|
|  Dasatinib|       564|
|Doxorubicin|       532|
|Vinblastine|       523|
|    Digoxin|       522|
+-----------+----------+
```
----------------
