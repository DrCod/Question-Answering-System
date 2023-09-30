# Question-Answering-System

This is a basic QnA machine learning system ( unfortunately still work under progess - failed to catch the deadline for completion ;) ).
It is supposed to carry out the following funtionalities:

* Text parsing : Ingest raw and unprocessed text corpus and parse it into structured form
* Embeddings : Extract input text and query text embeddings
* Retrieval : Take as input a text query and receive top 3 relevant passages from a given corpus
* Indexing : Create document index using [ElasticSearch](https://github.com/elastic/elasticsearch-py)
* GUI : An interactive user interface to handle querying

  # Get Started

  The simplest way to set up the application is to 'cd' into working directory and incrementaly execute scripts to generate results for executing subsequent functionality including interacting with elasticsearch instance

  # Start Parser
  
  python .\app\parsing.py --input_folder path\to\corpus --output_folder path\to\outputs --chunk_splits 5

  # Generate Embeddings
  
  python .\app\model.py --input_file path\to\input_csv --output_folder path\to\outputs --normalize_embeddings

  # Start Retrieval
  
  python .\app\retrieval.py --query "sample query" --index_name INDEX_NAME --n_returns 3 --password PASSWORD --host HOST_URL --output_folder \path\to\outputs --normalize_embeddings 
