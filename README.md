# Non-Stationary Inference
Code accompanying the project Non-stationary Inference of the Parameters of Theoretical Reinforcement Learning Models - Interpreting Human Behavior in MAB task

## Behavioral Dataset 
The behavioral dataset can be found in the paper *Models that learn how humans learn: The case of decision-making and its disorders - Dezfouli et al. (2019)*
https://figshare.com/articles/dataset/Models_that_learn_how_humans_learn_The_case_of_decision-making_and_its_disorders/8257259

## Artificial
- To generate a synthetic training-set run ```artificial/create_train_data.ipynb```. 
- To train a neural network model with the synthetic training-set run ```artificial/rnn_train.ipynb```.
- To fit both Baseline models run ```artificial/q_fit.py``` for Stationary Q-learning maximum-likelihood and ```artificial/fit_bayesian.py``` for Bayesian particle filtering.
- To create Fig 3 run ```artificial/fig3/draw_fig3.ipynb```
- ```trained_model.pth``` state_dict of the trained model weights used for the analysis. 
- ```trained_model_static.pth``` (static training-set) state_dict of the trained model weights used for the analysis. 



## Behavioral
- To generate a synthetic training-set run ```behavioral/create_train_data.ipynb```.
- To train a neural network model with the synthetic training-set run ```behavioral/rnn_train.ipynb```.
- To fit both Baseline models run ```behavioral/q_fit.py``` for Stationary Q-learning maximum-likelihood and ```behavioral/fit_bayesian.py``` for Bayesian particle filtering.
- To finetune the trained model with behavioral data run ```behavioral/finetune.ipynb```.
- For the PCA embedding and SVM classification run ```behavioral/classification.ipynb```.
- To create Fig 4 a and b and Fig 5 run ```behavioral/fig4_5/draw_fig4_5.ipynb```
- To create Fig 6 a, b and c run ```behavioral/fig6/draw_fig6.ipynb```
- ```trained_model.pth``` is a state_dict of the trained models weights used for the analysis (before finetune).


## Dependencie
- numpy
- pandas
- matplolib
- seaborn
- sklearn
- scipy 
- tqdm
- torch
