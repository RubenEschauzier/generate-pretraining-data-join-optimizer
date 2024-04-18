from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from tqdm import tqdm
import json


# In virtuoso.txt in database/
# MaxQueryCostEstimationTime 	= 400	; in seconds
def wrapper(url, default_graph):
    sparql = SPARQLWrapper(
        url
    )
    sparql.setReturnFormat(JSON)
    sparql.addDefaultGraph(default_graph)
    return sparql


def execute_query(query, wrapped_sparql_endpoint):
    wrapped_sparql_endpoint.setTimeout(60)
    wrapped_sparql_endpoint.setQuery(query)
    return wrapped_sparql_endpoint.queryAndConvert()


def count_results(result):
    count = 0
    bindings_found = []
    for r in result["results"]["bindings"]:
        count += 1
    return count


def execute_array_of_queries(queries, endpoint, ckp=None, queryRandom=True):
    query_strings = []
    query_cardinalities = []

    for i, query in enumerate(tqdm(queries)):
        if queryRandom:
            ret = execute_query(query[0], endpoint)
        else:
            ret = execute_query(query, endpoint)

        query_cardinalities.append(count_results(ret))
        if queryRandom:
            query_strings.append(query[0])
        else:
            query_strings.append(query)
        if ckp and i % 5 == 0 and i > 0:
            with open(ckp + "/query_strings.json", "w") as f:
                json.dump(query_strings, f)
            with open(ckp + "/query_cardinalities.json", "w") as f:
                json.dump(query_cardinalities, f)

    return query_strings, query_cardinalities


def test(endpoint, graph_uri):
    query = '''
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX snvoc: <http://www.ldbc.eu/ldbc_socialnet/1.0/vocabulary/> 
    PREFIX sn: <http://www.ldbc.eu/ldbc_socialnet/1.0/data/>
    SELECT ?messageId ?messageCreationDate ?messageContent WHERE {
        ?message snvoc:hasCreator sn:pers00000000000000000933;
        rdf:type snvoc:Post;
        snvoc:content ?messageContent;
        snvoc:creationDate ?messageCreationDate;
        snvoc:id ?messageId.
    }
    '''
    query2 = '''
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX snvoc: <http://www.ldbc.eu/ldbc_socialnet/1.0/vocabulary/>
    PREFIX sn: <http://www.ldbc.eu/ldbc_socialnet/1.0/data/>

    SELECT ?messageId ?messageCreationDate ?messageContent WHERE {
      ?message snvoc:hasCreator sn:pers00000004398046512167 ;
        snvoc:content ?messageContent;
        snvoc:creationDate ?messageCreationDate;
        snvoc:id ?messageId.
      { ?message rdf:type snvoc:Post. }
      UNION
      { ?message rdf:type snvoc:Comment. }
    }
    '''
    wrapped_endpoint = wrapper(endpoint, graph_uri)
    ret = execute_query(query2, wrapped_endpoint)
    print(count_results(ret))
    for r in ret["results"]["bindings"]:
        print(r)
