#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy of csci_49376_project_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
        https://colab.research.google.com/drive/1ybkA9zZjGQL2tuqIs8vPuQksRJDqSZ71
"""

import os
import sys
# Import SparkSession
from pyspark.sql import SparkSession

edge_file = "edges.tsv.gz"
node_file = "nodes.tsv.gz"

if 'COLAB_GPU' in os.environ:
    print("I'm running on Colab")
    # !pip install pyspark
    from google.colab import drive
    drive.mount('/content/drive')
    # drive.flush_and_unmount()
    edge_file = f"/content/drive/MyDrive/School/CSCI/Big-data/{edge_file}"
    # edge_file = "/content/drive/MyDrive/School/CSCI/Big-data/edges-test.tsv"
    node_file = f"/content/drive/MyDrive/School/CSCI/Big-data/{node_file}"
else:
    print("I'm running locally")
    import urllib.request
    nodes_gz_url = "https://drive.google.com/uc?id=1-VBD-Un8SRj6mmn38EQoacqRuBNizoi0&export=download"
    edges_gz_url = "https://drive.google.com/uc?id=1-7tacmfwahcRtN6mS9FKB8Ob_vbxgUxt&export=download"
    if not os.path.exists("nodes.tsv.gz"):
        print("Downloading nodes")
        urllib.request.urlretrieve(nodes_gz_url, "nodes.tsv.gz")
    if not os.path.exists("edges.tsv.gz"):
        print("Downloading edges")
        urllib.request.urlretrieve(edges_gz_url, "edges.tsv.gz")

# Create a Spark Session
ss = SparkSession.builder.master("local[*]").getOrCreate()
edges = ss.read.csv(edge_file, header=True, sep="\t")
nodes = ss.read.csv(node_file, header=True, sep="\t")

# print(f"Edges: {edges.count()}")
# print(f"Nodes: {nodes.count()}")

# sys.exit(0)

print("""
Q1. 

For each drug, compute the number of genes
and the number of diseases associated with the
drug. Output results with top 5 number of genes in a descending order
""")

# Filter out the Compounds associated with Genes & Diseases
disease_compounds = edges.filter(
        edges.source.startswith('Compound') &
        (
                edges.target.startswith('Disease')
        )
)

gene_compounds = edges.filter(
        edges.source.startswith('Compound') &
        (
                edges.target.startswith('Gene') 
        )
)

# gene_compounds.sample(.01).show(10)

# disease_compounds.show(10)

# Convert    |source|target::xyz| -> |source| 1 |
disease_rdd_1 = disease_compounds.rdd.map(lambda x: (x[0],    1) )
# print("Diseases: " , disease_rdd_1.take(3))
gene_rdd_1 = gene_compounds.rdd.map(lambda x: (x[0],    1) )
# print("Genes: ", gene_rdd_1.take(3))

disease_rdd_2 = disease_rdd_1.reduceByKey( lambda x, y: x + y)
# print("Diseases: " , disease_rdd_2.take(3))
gene_rdd_2 = gene_rdd_1.reduceByKey( lambda x, y: x + y)
# print("Genes: " , gene_rdd_2.take(3))

joined_rdd = gene_rdd_2.join(disease_rdd_2)
# joined_rdd.take(5)

sorted_joined_rdd = joined_rdd.sortBy(lambda x: x[1][0], ascending=False)
sorted_joined_rdd.toDF().show(3)

print("""
Q2:
Compute the number of diseases associated
with 1, 2, 3, …, n drugs. Output results with the top
5 number of diseases in a descending order.
E.g.
1 drug -> 2 diseases
2 drugs -> 1 diseases
""")
"""
```
❯ grep ^Compound edges.tsv| grep Disease:: | sort -k1 | awk '{print $1}' | uniq -c | sort -n | awk '{print $1}' | uniq -c
```
"""
disease_compounds = edges.filter(
        edges.source.startswith('Compound') &
        (
                edges.target.startswith('Disease')
        )

)
# disease_compounds.take(5)

rdd_1 = disease_compounds.rdd.map(lambda x: (x[0],    1))
# rdd_1.take(5)

rdd_2 = rdd_1.reduceByKey(lambda x, y: x + y)
# rdd_2.take(5)

df_1 = rdd_2.toDF(["compound", "drug_count"]).groupBy('drug_count').count()
df_1.sort("count", ascending=False).show(3)

print("""
Q3: Get the name of drugs that have the top 5 number of genes. Output the results.
```
MagicPill1 -> 2
MagicPill2 -> 1
MagicPill3 -> 0
```
""")
"""
```
cat edges.tsv| grep ^Compound:: | grep Gene:: | awk '{print $1}' | sort | uniq -c | sort -nr | head -5 | while read n c
do
    C=$(grep $c nodes.tsv | awk '{print $2}')
    echo $C $n
done
```

```
Crizotinib 585
Dasatinib 564
Doxorubicin 532
Vinblastine 523
Digoxin 52
```
"""

compound_nodes = nodes.filter(nodes.id.startswith('Compound'))
# compound_nodes.take(5)

rdd_1 = gene_compounds.rdd.map(lambda x: (x[0],    1))
# rdd_1.take(5)

rdd_2 = rdd_1.reduceByKey(lambda x, y: x + y)
# rdd_2.take(5)

rdd_4 = rdd_2.join(compound_nodes.rdd)
# rdd_4.take(5)

rdd_5 = rdd_4.sortBy(lambda x: x[1][0], ascending=False)
rdd_5.toDF().show(5)
