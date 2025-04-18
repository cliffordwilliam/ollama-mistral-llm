# Broken

Idk why the rendering does not work but you can copy paste this into another md viewer online it works fine


# AI study

## Deps

Need the following dependencies

```bash
pip install langchain
pip install -U langchain-ollama
pip install requests
```

Run the main.py

## What I have learnt and understood before doing this

# AI study

## LAM

LAM stands for Large Audio Model, It is an AI model trained on audio data to do audio tasks like speech recognition, speaker identification, and emotion detection

## LLM

Large Language Model. Text data trained to understand, generate, and manipulate human language.

## LVM

Large Vision Model. Trained with image or video data. To understand, generate, and reason visual content.

## Google Colab

A free, online coding environment by Google for Python. Used for data science and AI:

- Write **Python** code
- Free **GPUs/TPUs**
- Upload and access files
- **Share** notebooks

Note:

Why GPU and TPU?

- **GPU**
  - Was for rendering, but now it's good for data science or AI
  - So many cores run in parallel at once, which is good for numerous simple math operations!
- **TPU**
  - A better GPU
  - Made for AI uses
  - TensorFlow and multiplying huge arrays
  - Faster and more efficient than a GPU
  - Good for training and running models

How you use it in a nutshell:

1. Create a notebook
2. Save it by default to Google Drive
3. Write Python code similar to a Jupyter notebook

## LangChain

Python and JS framework to make an LLM app

Instead of sending one prompt at a time, LangChain:

- Chains multi-step!
- Integrate other tools like a web scraper
- Store memory like past chats
- Use search engines
- Manages agent behaviors

What are agents? Let’s talk about the following

With the prompt “What is 19.5% of 238?”

LLM tries, but it might be wrong; it does not access real-time info or remember past input

So with an agent, it uses LLM as its brain and has tools like calculators or file access. It may also have reasoning, logic, or memory of past chats. It can solve complex goals.

| **LLM alone** | **Agent** |
| --- | --- |
| 1 prompt = 1 answer | Completes the goal with reasoning |
| This is a tool | Uses many tools like LLM |
| Stateless, no memory | Has memory |
| Guesses | Step-by-step plan |
| Reactive | Proactive in completing a goal |
| Good for<br><br>“Summarize this text.” | Good for<br><br>“Read 3 files and tell me the trend of them, and send that as mail to me.” |

We developer makes the agent, we pick which LLM to use, tools to use, memory, and use a framework like LangChain.

But what exactly are they? I mean, what are LLM and Agent?

LLM

- This is a trained network.
- **STATELESS**, does not persist, cannot have super long conversations, passing previous sentences each time into it is not going to work as there is are token limit

But what is a network in this context?

A network here refers to 1 network that 1 or more physical machines can hold using their resources, AKA your GPU, to do the calculations that each node has to do. You use libraries to make these on your machines, like PyTorch. This is how people use it: they give inputs to it, then it gives us output. The idea is that the output should be closer to what we wanted, so any differences will affect each node (**backpropagation**). Specifically, it affects each node weight and bias that persists in memory (this is what we mean when you save a model). These weights and biases are what determine what the output is going to be like. Typically, the longer training you do, the closer the weight and bias are to making an output you want.

Also, when you create a neural network, the “brain” is just like any programming software; it takes up some memory you have in the CPU. Regardless, it just takes up CPU / GPU resources like any other software. You can think of each node like an object that holds some properties, which are the weight and the bias. The output is just numbers, but they differ on the layer that node is in (we arrange the nodes in layers); you can use different math functions, and each one has its own output shape, like maybe 0 to 1 or 0 to infinity. When you save a model, you save the whole weight and bias collection. Training is just updating the collection of weights and biases.

Note

- **Weight**:
  - If the weight for income is 0.8 and for age it's 0.2, then a change in income will have a larger effect on the output than a change in age, because the income feature has more influence (higher weight).
- **Bias**:
  - If the bias is set to 0.9, the model might require stronger inputs to classify something as True. If the weighted sum of inputs (like income and age) is lower than 0.9, the model will likely output False. But if the weighted sum exceeds 0.9, the model is more likely to output True.

Note that when you give input, different model has its own tokenization strategy where it breaks down the input into smaller pieces called token. How long or how the token looks like depends on the model strategy. Each token is processed by each node one by one where the node weight and bias adjust the data. The previous token iteration also affects the next output.

“””

import torch.nn as nn

model = nn.Linear(3, 2) # A simple layer with 3 inputs, 2 outputs

”””

Okay, but what about agents?

Agents are software, typically a web application, that uses tools and models to achieve a certain goal. Let’s say you want to make a web application to do the manual labour of booking a flight. You can leverage these tools:

- LLM
  - To understand the typed text
  - To also reply in a human text way
- API
  - Some API that does the transaction
- AND we have a way to make data persist! So we can have a long conversation with LLM or other models. We can maybe store that in the browser memory, say, using Angular for the frontend.

Okay, now what is **LangChain**?

It is a **framework for making an app with AI**, AKA agents:

- Uses LLM
- Uses tools
- Has multi-step reasoning
  - Meaning it can answer things like this: “Square root of Elon Musk's child’s age?”
  - It has to do with multi-step, gotta look for the child, find age, find the root.
  - Each step agent uses a tool, maybe an LLM for step 1, to read
  - LLM alone is 1 step, 1inputt, 1 output. It cannot duplicate actions
- Has chat memory

Note:

- When you use LangChain to make an app that uses LLM, people like to say **prompt templating**. That just means string templating for the input.t
- Specifically, a **reusable string template for LLM input**
- **It is just string templating you do all the time in programming, nothing more**

## Summary on Model, Agent, and LangChain

So models are referring to the network you can make with a framework or leverage 3rd party for it. If you want to build your own, you can use a framework like PyTorch. What is a model for? You want to give input and see the output it gives you. You want to keep doing this until the output is similar to what you desire. Say you train it to find cats in photos. This is what happens in a nutshell in a model:

1. You give input
2. That long input is tokenized; each model does it differently
3. You iterate each token, and per token, each node manipulates it
4. Note that each node is organized in layers, fyi
5. Manipulation of the token is affected by
    1. Node weight:
        1. Determines which argument has the most effect on the output
            1. If age has more weight, a small age difference can cause a huge output difference
    2. Node bias:
        1. Result should determine a boolean answer, like should customer buy or not buy a product. If you have a one sided bias like 0.9 then you should see answer that leans more toward one end. Maybe answers are more likely to have customer not buy. This is just an example output does not always have to be boolean.
6. When iteration is done, you get the result.
7. Result goes through backpropagation, this is the learning process where it back propagates to adjust each node bias and weight to make the answer to get closer to the desired answer.
8. Repeat from step 1 till you are happy with the model results, as it was able to find cats almost perfectly.
9. Then save the model. When you save a model, you save all the node weights and biases that are responsible for the results that you like

That is how the model works, but the model alone can only do 1 prompt, 1 answer. As it cannot do the following: find the child's name, or find his age. That is a 2-step thing that the model alone cannot do. The model alone is also stateless, as it does not persist information from the last sentence. You can make it kinda stateful by always piling up previous sentences to simulate a stateful conversation, but at one point, you will hit the token limit. Each model has a limit to how many tokens it can process at a time per 1 prompt! Imagine that it has to take a whole book's worth of length of text input, breaking all that down into tokens will take too much time or processing power, it's impossible.

Agents? These are apps of any form, mostly web apps. Used to achieve a certain goal, like helping people find flights with the use of models and tools:

- Models are usually LLM
  - It helps understand human input
- Tools?
  - It uses tools to achieve goals like browsing the net or making orders with the API

For example, you can use Angular for interface frontend, then use a framework like LangChain to integrate LLM and tools to create an app where people can type text into text box like help me find tickets to fly to Singapore. Then it uses one of the tools you integrate like maybe API to actually GET list of flights from a certain server somewhere. Then use Angular to render in browser.

That’s all about models, agents, and langchain.

## Trying out models locally for free

A solution is to use Ollama, I downloaded a pretrained model and an engine to run neural network locally and a interface from it. Model is called Mistral, and it is LLM. Models were trained by research companies, not the Ollama company, those guys just give us local access to LLM. Also the models are GGUF quantized versions, lower size at cost of precision. Specifically the data that the smaller version uses use smaller bits. GGUF is just the file format.

Note about Ollama. They focus on convenience. Everyone has access to local LLM more easily. That is their business model.

“””

Windows PowerShell

Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! <https://aka.ms/PSWindows>

> ollama --version

ollama version is 0.6.5

> ollama list

NAME ID SIZE MODIFIED

> ollama pull mistral

pulling manifest

pulling ff82381e2bea... 100% ▕████████████████████████████████████████████████████████▏ 4.1 GB

pulling 43070e2d4e53... 100% ▕████████████████████████████████████████████████████████▏ 11 KB

pulling 491dfa501e59... 100% ▕████████████████████████████████████████████████████████▏ 801 B

pulling ed11eda7790d... 100% ▕████████████████████████████████████████████████████████▏ 30 B

pulling 42347cd80dc8... 100% ▕████████████████████████████████████████████████████████▏ 485 B

verifying sha256 digest

writing manifest

success



”””

Check again to see if the model is in the machine

This shows that it does have the model, it is a Mistral LLM pre-trained

“””

> ollama list

NAME ID SIZE MODIFIED

mistral: latest f974a74358d6 4.1 GB 5 minutes ago



”””

Great, but what has Mistral been trained for anyway? It is pretrained to guess the next token. Say you input a string sentence. Then it will use its unique tokenization to break it down into smaller tokens. Per token, it will try to guess the next token based on the previous token, if there are any for context, and also the weight and bias of each node/neuron that has been trained. Which should create a human reply, like an understandable human reply, AKA talking. Or rather, generate language based on each token to be more precise.

Create a repo dir, make a venv in there. I am using Python since there is a library for the LangChain and Ollama.

python -m venv .venv

I also pushed this to the remote as well.

Output at the time of making looks like this from agent

```bash
python .\main.py
LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.        
  agent = initialize_agent(


> Entering new AgentExecutor chain...
 To get information about Pikachu, I will use the `PokeAPIFetcher` tool. The name of the Pokemon is "Pikachu".

Action: PokeAPIFetcher
Action Input: "Pikachu"
Observation: pikachu has a height of 4 decimetres.
Stats: {'hp': 35, 'attack': 55, 'defense': 40, 'special-attack': 50, 'special-defense': 50, 'speed': 90}
Thought: I now know the final answer.
Final Answer: Pikachu has a height of 4 decimetres and its stats are {'hp': 35, 'attack': 55, 'defense': 40, 'special-attack': 50, 'special-defense': 50, 'speed': 90}.

> Finished chain.
{'input': 'Tell me about Pikachu including its stats.', 'output': "Pikachu has a height of 4 decimetres and its stats are {'hp': 35, 'attack': 55, 'defense': 40, 'special-attack': 50, 'special-defense': 50, 'speed': 90}."}
```
#   o l l a m a - m i s t r a l - l l m 
 
 
