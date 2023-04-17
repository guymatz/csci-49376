## To Run:

1. `pip install -r requirements.txt`
2. `python p2.py`

NB: No datafiles are required.  They will be downloaded if they are not available locally (as gzip'ed files)

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
Using a dataset of graph edges, two RDDs were created,
one of compounds and associated diseases, and another of compounds and associated genes.

#### Pseudocode
```
class Mapper
    method Map(string t, integer r)
        Emit(string t, pair (r, 1))

class Reducer
    method Reduce(string t, paris [(s_1, c_1),(s_1, c_1)...])
        sum <- 0
        cnt <- 0
        for all pair (s,c) \in pairs
            sum <- sum + s
            cnt <- sum + s
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
* Join

#### Approach
Using a dataset of graph edges, two RDDs were created,
one of compounds and associated diseases, and another of compounds and associated genes.

#### Pseudocode
```
class Mapper
    method Map(string t, integer r)
        Emit(string t, pair (r, 1))

class Reducer
    method Reduce(string t, paris [(s_1, c_1),(s_1, c_1)...])
        sum <- 0
        cnt <- 0
        for all pair (s,c) \in pairs
            sum <- sum + s
            cnt <- sum + s
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
Using a dataset of graph edges, two RDDs were created,
one of compounds and associated diseases, and another of compounds and associated genes.

#### Pseudocode
```
class Mapper
    method Map(string t, integer r)
        Emit(string t, pair (r, 1))

class Reducer
    method Reduce(string t, paris [(s_1, c_1),(s_1, c_1)...])
        sum <- 0
        cnt <- 0
        for all pair (s,c) \in pairs
            sum <- sum + s
            cnt <- sum + s
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
