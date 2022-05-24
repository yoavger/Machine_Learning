# Non-Stationary Inference
Code accompanying the project Non-stationary Inference of the Parameters of Theoretical Reinforcement Learning Models - Interpreting Human Behavior in MAB task

## Artificial
- To genrate a synthetic training-set run  ```artificial/create_train_data.ipynb```. 
- To train a neural netwrok model run  ```artificial/rnn_train.ipynb```.
- To fit both Baseline models run ```artificial/q_fit.py``` for Stationary Q-learning and ```artificial/fit_bayesian.py``` for Bayesian particle filtering.
- ```trained_model.pth``` state_dict of the trained model weights used for the analysis. 
- ```trained_model_static.pth``` (static training-set) state_dict of the trained model weights used for the analysis. 
- To creat Fig 3 run ```artificial/fig3/draw_fig3.ipynb```


## Behavioral
- To genrate a synthetic training-set run ```behavioral/create_train_data.ipynb```.
- To train a neural netwrok model run ```behavioral/rnn_train.ipynb```.
- To fit both Baseline models run ```behavioral/q_fit.py``` for Stationary Q-learning and ```behavioral/fit_bayesian.py``` for Bayesian particle filtering.
- To finetune the train model with behavioral data run ```behavioral/finetune.ipynb```.
- For the PCA embedding and SVM classification run ```behavioral/classification.ipynb```.
- ```trained_model.pth``` is a state_dict of the trained models weights used for the analysis (before finetune).
- To creat Fig 4 a and b and Fig 5 run ```behavioral/fig4_5/draw_fig4_5.ipynb```
- To creat Fig 6 a, b and c run ```behavioral/fig6/draw_fig6.ipynb```

## Dependencie
- numpy
- pandas
- matplolib
- seaborn
- sklearn
- scipy 
- tqdm
- torch
