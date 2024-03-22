# Evaluation

## HellaSwag

[HellaSwag](https://rowanzellers.com/hellaswag/) is a challenge dataset for evaluating commonsense inference. It consists of **70k** multiple choice questions about grounded situations.

Sample question from HellaSwag.

#### Example 1

**Context**
A tray of potatoes is loaded into the oven and removed. A large tray of cake is flipped over and placed on counter. A large tray of meat _____.

**Options**
A. is poured into a midden.
B. is placed onto a baked potato.
C. and pickles are placed in the oven.
D. is prepared then it is removed from the oven by a helper when done.


#### Example 2

**Context**
We see a lady washing a white towel in a sink. The lady lifts the towel up. The lady picks something up off the floor. The lady ____. 

**Options**
A. puts the towel on a cabinet and ties it.
B. then lifts up the towel.
C. throws the item she just washed at the sink.
D. wrings out the towel in the sink.


The [HellaSwag paper](https://arxiv.org/abs/1905.07830) shows that commonsense inference was difficult for state-of-the-art models by the time that the paper was published. These questions are trivial for humans (>95% accuracy), but only 48% accuracy for the state-of-the-art models at that time.

More recent LLM models now can achieve much higher accuracy on the HellaSwag datasets. For example, ChatGPT reaches 85.5%, and GPT-4 reaches 95.3%. A list of models (paper) and their performance on the HellaSwag dataset can be found [here](https://paperswithcode.com/sota/sentence-completion-on-hellaswag).