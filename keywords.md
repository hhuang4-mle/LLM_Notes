
## Transfer Learning

The idea of transfer learning is to take a pre-trained model and use it (adapt it) to a new task. Because pre-training the model is quite expensive, transfer learning  exploits the knowledge gained from a previous task to boost the performance on a related task with lower costs.

## Truncation & Padding

Batched inputs are often different lengths and can’t be converted to fixed-size tensors.
Truncation and Padding are strategies for make the text input the same length.

* Padding adds a special padding token to extend shorter sequences to a fixed length.
* Truncation works in the other direction by truncating long sequences to a fixed length.


## Hallucinations 

LLM models generate plausible but actually incorrect or nonsensical information.
Possible reasons:

1. Incomplete or Noisy Training Data
2. Vague Questions
3. Absence of Grounding
4. Lack of “common sense” reasoning abilities


## Catastrophic Forgetting (Catastrophic interference)
* When you feed data to a model, new pathways are formed, sometimes causing the algorithm to “forget” the previous tasks it was trained for. Sometimes, the margin of error increases, but other times, the machine completely forgets the task.
* Strategies to minimise the issue.
	* Create backups before re-training a model
	* Train a new model with both new and old data together

