## Learning Objectives 

This introductory level workshop aims to get participants comfortable with tools like jupyter notebooks and LlamaIndex  as a jumping off point for further self-guided exploration.

## Tutorial Project

![demo](https://media0.giphy.com/media/xUA7b12v8mfc37Ekus/giphy.gif?cid=ecf05e470adx2svalo5zvpm6yhybru4pnoxbjy1is6khh633&ep=v1_gifs_search&rid=giphy.gif&ct=g)


We're going to build a toy [RAG](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html#retrieval-augmented-generation-rag) question-answering bot that is specialized for giving us plot summaries of the first season of Better Call Saul. We'll see how easy LlamaIndex makes it to create a RAG bot that downloads information from the internet and packages that as context for the bot.

Ultimately, we want to build a bot that correctly answers the following question:

> "What happened in episode 3 of season 1 of better call saul? Summarize the episode in bullet points.

## Setup

There are a few setup steps below before you can get to the good stuff, however once you set everything up once, you'll have a blueprint for experimentation going forward.

### Prep

1. Make sure you have python 3 installed.
Consult [this doc](https://docs.python-guide.org/starting/install3/osx/) for details if you think it's possible you only have python 2.
1. You will need access to your own `OPEN_API_KEY`. Many LLM tools expect this key to exist in your env. To obtain a key, you will need a personal account with Open AI that has some credit on it. $5 should be plenty. Make sure to **turn off auto recharge** so that you aren't unexpectedly billed more. Once you've signed up and put money in your account, learn about how to [add the key as an environment variable here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
2. Make sure you have acess to global installs for the following python libs:
   1. jupyter notebooks:`pip install notebook`
   2. virtualenv: `pip install virtualenv`
3. Fork this repo
4. Read through the [llamaindex starter tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html).

### 2. Create a virtualenv
1. Here's a little background on why you want to [use a virtualenv](https://www.zainrizvi.io/blog/jupyter-notebooks-best-practices-use-virtual-environments/).
2. Make sure you're in the `tutorials` folder after cloning this repo.
3. Create the virtualenv: `python -m venv myenv`
4. Active the virtualenv: `source myenv/bin/activate` 
5. [For later] To deactivate the virtualenv once you are done: `deactivate`

### 3. Open the starter notebook
1. Open vscode:  `code .` [(More details about this command)](https://code.visualstudio.com/docs/setup/mac#_launching-from-the-command-line)
2. Open `tutorial/better-call-saul/starter-notebook.ipynb`.
3. Select `Python environments/myenv` as the kernel for the notebook (you might be promptyed to install `ipykernel`).

### 4. Basic setup
  First, we'll use LlamaIndex to set up the simplest possible question answering bot, using no external data sources. Spend some time reading the notebook `starter-notebook.ipynb` and, once you're ready, sequentially run the cells. At the bottom, you will probably see a response that includes some degree of hallucication and/or a refusal to answer the question:

 ```
 I'm sorry, but as an AI language model, I don't have real-time data or access to specific TV show episodes. However, you can find a detailed summary of episode 3 of season 1 of Better Call Saul on various TV show databases or streaming platforms.
 ```
  
### 5. Adding RAG
In order to get quality results, we need to add more context on the first season of Better Call Saul. We can do that by creating an actual RAG system as intended by LlamaIndex instead of the stub one we just ran in the starter notebook. The rest of the tutorial we will spend updating the notebook to produce a better response.

First up, we need to actually aquire some reliable data to work off of. I have provided a list of wikipedia urls for each episode from the first season of the show in `season_one_episodes.json`. Now we just need to turn them into downloaded documents ready to be indexed.

1. First, add a new cell with the following code underneath the cell that sets up tracing (cell 2). (Order is important!):

This cell updates llamaindex to use a Huggingface model for embedding instead of OpenAI, saving us a little cash:

```python
# to save money we're using the local embed model instead of the OpenAI default text-embedding-ada-002
# this will use a HuggingFace embedding model instead

from llama_index import (
    ServiceContext,
    set_global_service_context,
)

service_context = ServiceContext.from_defaults(embed_model="local")
set_global_service_context(service_context)
```


1. Next, replace the cell with the comment `#stub out the index for demo purposes` with the following content:

```python
import json
import os.path

from llama_index import (
    StorageContext,
    VectorStoreIndex,
    download_loader,
    load_index_from_storage,
)

def create_and_store_embeddings():
    # load wikipedia urls
    f = open("seasion_one_episodes.json")
    urls = json.load(f)

    # -----TODO:-----
    # download data from the urls in the urls array, and parse it into documents
    # Hint: look in the llamahub for an 
    # appropriate data loader: 
    # https://llamahub.ai/?tab=loaders
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_existing_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)


PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    index = create_and_store_embeddings()
else:
    index = load_existing_index()
```

This cell is where the downloading, parsing and indexing of our Better Call Saul data occurs.

3. Finally, **replace** the final cell in the notebook with this code:

```python
text_qa_template_str = """
    #  ----TODO:---- add context information here
     Using only the context information, answer the question: {query_str}
     If the context isn't helpful, say that you don't know the answer."""

rag_text_qa_template = PromptTemplate(text_qa_template_str)

has_rag_context_query_engine = index.as_query_engine(
    text_qa_template=rag_text_qa_template
)

response = has_rag_context_query_engine.query(base_prompt)
print(response)
```

As you might have noticed, there are two places in the cells we added that have `TODO` tasks for you to do in order to get this working. We need to:

1. choose an appropriate data loader for the wikipedia data, and 
2. edit the `text_qa_template_str` to embed the context_str. Take some time to play around with the code to see what you can whip up. 
   
**HINT:** Use the Arize tracing interface to check out what is happening under the hood and see if you're getting closer.

<details>
  <summary>A solution for loader</summary>
 
```python
BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=urls)
```

There are multiple solutions to this one, including using the dedicated Wikipedia loader.

</details>

<details>
  <summary>Solution for context string in prompt</summary>

 ```python
text_qa_template_str = """
  Context information is provided below:
  ________________________________________
  {context_str}
  ________________________________________
  Using only the context information, answer the question: {query_str}
  If the context isn't helpful, say that you don't know the answer.
  """
```

</details>


### 6. Getting it all working

Now, when running the notebook, you should get an answer that looks like this:

<details>
  <summary>Spoiler alert</summary>

- The episode begins with a flashback to 1992, where Chuck McGill meets Jimmy McGill in jail and agrees to represent him if he stops running cons and finds legitimate employment.
- In the present, Jimmy is anxious about Nacho Varga's plot to steal from the Kettlemans and warns Kim Wexler about the potential danger.
- Jimmy anonymously calls the Kettlemans and warns them, leading them to see someone surveilling them from a van.
- The next morning, Jimmy finds out that the Kettleman family is missing and believes Nacho has kidnapped them.
- Jimmy is picked up by the police and learns that Nacho has been arrested and requested Jimmy as his lawyer.
- Nacho admits to surveilling the Kettlemans but denies kidnapping them and warns Jimmy that he will have him killed if the charges are not dropped.
- Jimmy convinces Kim to take him to the Kettleman house to investigate and notices inconsistencies, suggesting that the Kettlemans staged their kidnapping.
- At the courthouse, Jimmy starts a fight with Mike Ehrmantraut, who subdues him. The police ask Mike to press assault charges, but he changes his mind after hearing Jimmy's theory about the Kettlemans' disappearance.
- Mike suggests that the Kettlemans are hiding somewhere close to home, and Jimmy explores the desert near their house and discovers their stolen money.

</details>


Much better! You can clone this notebook and use it as a jumping off point to build your own RAG experiments. 

