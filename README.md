# GoT RAG Server

## Running the application
(Docker required)

```
docker build -t norm-ai-takehome . &&

docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:80 norm-ai-takehome
```

(No Docker option)
```
uv sync && uv run app/main.py
```

Navigate to http://localhost:8000/docs for API Documentation.
Hit "Try it out" and enter your query to play with the application

## Architecture

The two main pieces to the application are the DocumentService and QdrantService. 

`app/document.py`
The DocumentService parses pdf documents, looks through sections and subsections, and labels them heirarchically. An open todo would be to relate them heirchically as Documents in the database

`app/qdrant.py`
The Qdrant Service sets up a Qdrant vector database locally, and queries the parsed documents. I increased the "k" value because I saw better results this way. This reconstructs the original text placement based on metadata from each of the documents.


## Assumptions and Caveats
With more time, I think a small dev qdrant instance or a locally running qdrant docker container
and a docker-compose setup would provide a more realistic local dev exeprience. 
This was also my first time using Qdrant and Llama Index, so it took me a bit of time to familiarize myself with the documentation.
With more time, I would focus on the document ingestion. Having well labeled and structured inputs will immensely help with the querying quality.

## Reflection

>Q: What unique challenges do you foresee in developing and integrating AI regulatory agents for legal
>compliance from a full-stack perspective? How would you address these challenges to make the system
>robust and user-friendly?

Legal and compliance workflows demand precision. There's no room for letting hallucinations slip
into the product. This introduces constraints on how products are built. 

The greatest challenges that I foresee are ensuring precision and correctness, the durability of long running AI agents, and the complexity of integrating AI agents and workflows into business processes, security requirements, and regulatory requirements.

### Precision and Correctness
In an environment where mistakes are especially costly, verification of any AI completed work, or AI generated content is critical.
As shown in this take-home, citation of original source material is one way to verify the correctness of AI work. I think the near term future of AI agents in the legal/compliance space involves human-in-the-loop workflows, where human experts are augmented by AI assistants. The dream of fully autonomous agents may be some time in the future.

Something interesting to explore that may help make autonomous agents a reality could be formal verification of AI work. Is there a way to formally verify AI completed work in the same way that you can write proofs to verify correctness of code?
If AI agents operate "as code", then there may be ways to prove certain behaviors that they take.

### Durability of Long Running Agents
One of the main engineering problems with AI agents concerns general infrastructure durability and reliability, as well as efficient management of AI agents context windows. Any long running AI agent will need ways to continue operating in the face of various infrastructure outages from dependencies. LLM providers aren't world renowned for their uptime and AI agents may have to be integrated into internal systems for clients. What do you do when any of these components have downtime? Any system thinking about AI agents needs to be thinking about fallbacks, long running workflows, agents being able to start and stop based on outage state and much more. This means architecting systems from the ground up with reliability and durability in mind. "OpenAI's API is down" may not be a valid excuse for agent downtime.

### Integration Complexity
Operating in the legal and compliance requires that agents also comply with any government regulation or security constraints. Imposed on them. This leads to novel infrastructure challenges. Agents may have to operate on client infrastructure and integrate with client systems, while respecting their security and compliance constraints for software vendors.  This can make observability, and verification of agent work much more difficult. Ensuring the agents are flexible, and able to run on a variety of infrastructure solves many of these headaches. Systems should be built ready to be multi-cloud and even potentially on-prem from the design phase. Agents should also ship with extensive observability, that allow for diagnostics and verification of work, while respecting client security and privacy.


