# Forex_Prediction

The goal of this project is to predict market price of forex data using SVM function with preprocessed features including technical indicators and price action signals. These features can be implemented and edited using config files without any coding invovled, best for people without coding knowledge.

The main goal was to use this tool to predict binary action of price value in the next X candles, which means if value is going to be higher or lower in the future.

There are two ways to run the code, one is using step_svm which runs svm on a small window size and shift the window as data goes forward.
The second way is to run the svm on the whole dataset and predict the future.
