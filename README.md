Text Classifier using TensorFlow
================================

- [Code](TextClassifier.ipynb)
- [Project Description](#project-description)
- [What I Learned](#what-i-learned)
- [Project Highlights](#project-highlights)
- [Things to Improve](#things-to-improve)


Project Description
-------------------

This project is centered around the development of a Text Classifier using ``TensorFlow``, aimed at categorizing text messages into 'spam' or 'ham' (non-spam). The motivation behind this project was to learn more about Natural Language Processing (NLP) and understand how machine learning models can be trained to distinguish between different types of text data.

### What I Learned

*   **TensorFlow and Neural Networks**: Gained hands-on experience with TensorFlow, exploring its capabilities to build and train neural networks. Learned about different layers and architectures suitable for NLP tasks.
*   **Data Preprocessing for NLP**: Understood the importance of preprocessing text data, including tokenization, padding, and encoding, to convert raw text into a format that can be fed into a neural network.
*   **Model Evaluation and Optimization**: Learned how to evaluate the performance of a model using metrics such as accuracy and loss, and experimented with different model architectures and hyperparameters to improve performance.
*   **Practical Challenges**: Encountered and overcame various challenges, such as overfitting, underfitting, and making the model generalize well to unseen data.

### Project Highlights

*   **Model Architecture**: The model features a neural network, whose first layer is a ``Text Vectorization layer`` (like a tokenizer) followed by a set of ``bidirectional layers`` . This architecture was chosen for its ability to understand the semantic meaning of text. The final layer is a single output. A number between 1 and 0, indicating how likely our message is to be spam.

*   **Dataset**: Utilized a labeled dataset of text messages, which was crucial in training the model to distinguish between spam and ham messages accurately. We "manually" tokenize the binary output, and use a ``Text Vectorization layer`` to tokenize any text input. 
*   **Results**: Achieved promising results in terms of accuracy and performance, indicating the model's effectiveness in classifying text messages.

### Things to Improve

This project was a significant learning experience, providing insights into the complexities of working with text data and the power of TensorFlow in building sophisticated machine learning models. It highlighted the importance of a well-thought-out preprocessing pipeline and the need for careful model tuning to achieve high accuracy in classification tasks.

However, there are a number of things that could be improved for future projects. 

*   **Expand the Dataset**: To improve the model's robustness and ability to generalize, I plan to train it on a more diverse and larger dataset.
*   **Experiment with Different Models**: Exploring other model architectures, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), to enhance classification performance.
*   **Deploy the Model**: Implement a web or mobile application that uses the trained model to classify messages in real-time. This allows direct user inputs, and implementation with other applications. 