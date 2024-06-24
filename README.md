# ForecastPro 📈
* A comparative study of Deep Learning Based Time Series models 

[![v0 Release](https://img.shields.io/badge/version-v0.0-blue)](#)

## Overview

1. **Models Implemented**:
   - **Stacked LSTMs**:  Captures long-term patterns in time series data using multiple LSTM layers for enhanced learning of temporal dependencies.
   - **LSTNet**: Combines CNNs and LSTMs to efficiently capture both short-term and long-term patterns in time series data, outperforming traditional stacked LSTMs.
   - **ConvLSTMs**: Integrates convolutional layers with LSTMs to capture spatiotemporal dependencies, making it suitable for tasks like air quality prediction.

2. **Key Advantages**:
   - **Stacked LSTMs**:
     - Enhanced ability to learn long-term dependencies.
     - Improved modeling of intricate temporal patterns.
   - **LSTNet**:
     - Combines CNNs and LSTMs for better feature extraction and pattern learning.
     - Addresses vanishing gradient problem common in deep recurrent architectures.
   - **ConvLSTMs**:
     - Captures both spatial and temporal dependencies.
     - Effective for applications involving spatiotemporal data, such as environmental monitoring.

3. **Comparison with Statistical Models**:
   - Statistical models (e.g., ARIMA, ARCH) rely on predefined assumptions and smaller datasets, offering interpretable coefficients and computational efficiency.
   - Deep learning models, including LSTNet and ConvLSTMs, require more data and computational resources but offer superior performance in capturing complex patterns.

4. **Key Features**:
   - **Side-by-Side Model Comparison**: 🔍 Predictions from Stacked LSTMs, LSTNet, and ConvLSTMs can be compared side by side, allowing for comprehensive evaluation of model performance.

5. **Advancements and Results**:
   - LSTNet and ConvLSTMs show promising results in addressing traditional challenges of deep learning models in time series forecasting, demonstrating improved accuracy and performance.

For detailed information and implementation, refer to our comprehensive articles and resources listed in the [References](#references) section.

## LSTNet(iterated forecast) 
   Apple stock price(data used till June 30th 2023)
   Index 3000+ is the forecast
   ![image](https://github.com/dsgiitr/ForecastPro/assets/103068685/07475ff5-ede7-439e-b23a-c3cb8a40b32a)

## ConvLSTM On AirQuality UCI.
   ### PAST CONFIDENCE IS THE PERFORMANCE OF THE MODEL ON PAST DATA
   ![image](https://github.com/dsgiitr/ForecastPro/assets/103068685/5764ddc2-1040-46ef-b567-a500b36db286)
   ### FORECAST
   ![image](https://github.com/dsgiitr/ForecastPro/assets/103068685/f373b544-b2dc-44b3-9b1b-2df9f51365f6)

## LSTNet ARCHITECTURE
(![LSTNet Overview](image.png))

This repository provides an implementation of stacked LSTMs and LSTNet, two deep neural network models designed to capture long- and short-term temporal patterns in time series data. This readme file aims to provide a comprehensive understanding of these models, their advantages, and why LSTNet outperforms stacked LSTMs. Additionally, it includes a brief description of statistical autoregressive models and the reasons why deep learning (DL) models have traditionally underperformed them.

## Stacked LSTMs

Stacked Long Short-Term Memory (LSTM) networks are an extension of the LSTM architecture, a type of recurrent neural network (RNN). Stacked LSTMs are employed to capture complex temporal dependencies and improve the model's ability to learn long-term patterns in sequential data.

In a stacked LSTM, multiple LSTM layers are stacked on top of each other to form a deeper neural network. Each LSTM layer receives the hidden state output from the preceding layer as input. By incorporating more LSTM layers, the model can learn hierarchical representations of the input sequence, with each layer learning at different levels of abstraction.

Benefits of using stacked LSTMs include:
- Enhanced capacity to capture long-term dependencies in the data.
- Improved ability to learn intricate temporal patterns.
- Effective modeling of sequential data for tasks such as time series forecasting and sequence generation.

## LSTNet

LSTNet (Long- and Short-Term Time-series Network) is a deep learning model specifically designed for time series forecasting. LSTNet combines the strengths of Convolutional Neural Networks (CNNs) and LSTMs to capture both short-term and long-term patterns in the data.

LSTNet comprises two main components: a CNN-based encoder and an autoregressive decoder. The CNN encoder extracts local patterns and features from the input time series data, enabling efficient feature extraction and capturing short-term dependencies. The extracted features are then fed into the LSTM decoder, which models long-term dependencies and generates accurate predictions for future time steps.

Key advantages of LSTNet over stacked LSTMs:
- Effective combination of CNN and LSTM layers to capture short-term and long-term dependencies, respectively.
- Improved feature extraction capabilities, allowing for better representation learning.
- Addressing the vanishing gradient problem, which is common in deep recurrent architectures.

## Why Deep-Learning Based Time Series Models?

### 1. **Handling Complex Patterns**

Deep learning models, such as ConvLSTM, Stacked LSTM, and RNNs, excel at capturing complex and non-linear patterns in time series data that traditional statistical models might miss. These models can learn intricate temporal dependencies and relationships that are essential for accurate forecasting in many real-world applications.

### 2. **Feature Learning**

Deep learning models automatically learn relevant features from raw data without the need for manual feature engineering. This is particularly beneficial for time series data, where identifying the right features can be challenging and time-consuming. Models like ConvLSTM combine convolutional layers with LSTM layers to extract spatial and temporal features simultaneously.

### 3. **Scalability**

Deep learning models are highly scalable and can handle large datasets effectively. This is crucial for applications involving big data, where traditional statistical models might struggle with computational efficiency and performance.

### 4. **Flexibility and Adaptability**

Deep learning models can be adapted to various types of time series data, including non-stationary and highly non-linear data. They do not rely on strong assumptions about the data distribution, making them more flexible than statistical models.

### 5. **Robustness to Noise**

Deep learning models, particularly those with multiple layers and complex architectures, are more robust to noise in the data. They can learn to filter out irrelevant variations and focus on the underlying patterns, leading to more reliable predictions.

### 6. **Transfer Learning and Pre-trained Models**

The availability of pre-trained models and the concept of transfer learning in deep learning allow for the application of knowledge gained from one domain to another. This can significantly reduce training time and improve performance, especially when dealing with limited data in specific time series applications.



## Challenges Faced by DL Models

DL models have traditionally encountered challenges in outperforming statistical autoregressive models for time series forecasting due to several reasons:

1. **Limited data**: DL models require a large amount of data to generalize well and learn complex patterns. In contrast, statistical autoregressive models can perform reasonably well with smaller datasets.

2. **Feature engineering**: DL models often rely on manual feature engineering to extract relevant features from raw time series data. In contrast, statistical autoregressive models incorporate domain-specific knowledge and assumptions into the model design.

3. **Interpretability**: Statistical autoregressive models provide interpretable coefficients and statistical tests, making it easier to understand the underlying dynamics of the time series. DL models typically lack this level of interpretability.

4. **Computational efficiency**: DL models tend to be more computationally demanding compared to statistical autoregressive models, especially for large-scale datasets.

However, advancements in DL techniques, such as LSTNet, have shown promising results in addressing these challenges and outperforming traditional statistical autoregressive models in certain scenarios.

<!-- ## Results -->

<!-- Placeholder for the results of the project. -->

<!-- ## Directions of Use -->

<!-- Placeholder for the updated directions of use as the project proceeds. -->

### About us
You can find more of such projects at: https://dsgiitr.in/

## References

The following references provide more information about the concepts and techniques used in this project:

1. "Stacked LSTM Networks" by Towards Data Science:
   [Link to the article](https://medium.com/towards-data-science/stacked-long-short-term-memory-networks-4e3fc9f807d2)

2. "LSTNet: A Deep Learning Model for Time Series Forecasting" by Medium:
   [Link to the article](https://medium.com/analytics-vidhya/lstnet-a-deep-learning-model-for-time-series-forecasting-28550fb9c3c8)

3. "Introduction to Statistical Autoregressive Models" by Medium:
   [Link to the article](https://medium.com/@josephruffianto/introduction-to-statistical-autoregressive-models-7f1b6b7fcfb3)

4. "Why Deep Learning Lags Behind Traditional Statistical Models in Time Series Forecasting?" by Medium:
   [Link to the article](https://medium.com/@aakashns/why-deep-learning-lags-behind-traditional-statistical-models-in-time-series-forecasting-15e7e2d89cf)

Please refer to these articles for in-depth explanations and insights into the concepts discussed in this README.

