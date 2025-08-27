# Stream of Consciousness Notes

## Setup
Got OpenAI api key set up, reading through Qdrant docs and 
llama index.

Most of the focus will be on getting really good, enriched 
docs in the input parsing. Breaking down into sub sections
and tying everything together with metadata seems correct. 
In a production implementation, I would need a way to link
documents by their section/subsection as law is heavily 
"hyperlinked". 

Going to run through some planning with strong models, and 
then start on the document model and parsing.

## Document parsing
using pymupdf seems to work well.
Splitting into subsections with metadata on each section
so that we can properly site things. The tests work well

## Querying (Qdrant Service)
Got it to run locally and return a result pretty quickly
after catching the little typos. Now I need to improve the 
actual returned citations. Right now they say source 'Unknown'. 
I need them to return the actual subsection.


## Running Locally
