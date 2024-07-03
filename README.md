This repository contains code to: 
1. Embed entities in a graph database to vectors using either `onehot-encoding` or `rdf2vec`. The rdf2vec implementation ensures all entities with a possible random walk of the specified depth will be generated.
2. Randomly generate queries from the graph in: `Path`, `Star`, and `Complex` shapes. These queries are generated using weighted random walks, assigning a higher probability of choosing a predicate to walk to if it is used less.
   Thus, preventing the random walk from walking over common predicates only. Furthermore, from these random walks, we generate random walks with some predicates randomly swapped, simulating empty queries.
   The resulting random walks are converted to queries (which can contain literals / named nodes), and some literals or named nodes are replaced by random entities, thus generating additional empty queries.
   Finally, we filter all isomorphicly equivalent queries.
3. Execute the generated / input queries (in the default case WatDiv queries) on a virtuoso endpoint and record the query string and its associated cardinality.

The resulting embeddings and query-cardinality pairs can be used to train a cardinality estimation model.
