# Aurora

## Introduction
Aurora is a powerful large language model optimized to efficiency and low computation. While leveraging the transformer architecture this model is made possible. Aurora is trained on large datasets, and then fine-tuned to produce useful answers. As a low computation and open-source model, Aurora simplifies LLMs for everyone, both in training and in utilisation. (NOTE: Aurora is based on a highly mathematical algorithms which trains on data, given this, Aurora can make mistakes and it is not fully perfect and reliable.)

## How it Works
While explaining the architecture behind Aurora is very complicated, here is a brief comprehension. The Aurora language model is based on a decoder only transformer, specifically, it somewhat follows the gpt-2 architecture. First the block goes through tokenization, this is a way of representing text as digits. Then the tokens pass through an embedding layer, this assigns intelligent integers to every word, words that are similar, also have similar embedding values. Simultaneously, the block goes through positional embedding to maintain integers which take the word structure into account. The embedding and positional embedding values are added together to make the input block. This input block then goes through layer normalization. The normalized block goes through multi-head-self-attention, this represents the block as an integer on how different words relate, it helps determine the long term interactions between words. After this, the output block goes through layer normalization once again. These newly normalized values pass through a two layer feedforward network with a gelu activation function. The final output, is then fed into a feedforward nerual network determining the next word. This is the generation, the training involves then determining the loss of this network based on the prediction, and backward propagating it in a sequential manner through the transformer. After this network is fully trained, it will have the capability to generate cohesive human-like text. The next step is applying fine-tuning so that it becomes a conversational chat-bot. (NOTE: This process is repeated in flexible ways depending on hyper parameters, for example the number of layers, or number of epochs. This explanation only represents a brief outline disregarding hyper-parameters)



![Alt text](diagrams/decoder.png)

## Conclusion
Aurora leverages state-of-the-art technologies for word generations. It utilizes the gpt-2 paper in order to determine optimal yet efficient low-weight parameters. Although it does modify gpt-2 to achieve more powerful results. Aurora is yet not available for the public, it is currently being implemented and trained. While Aurora is not the strongest model, Sociloc has plans for expanding computation power and implementing a greater language model.
