
# Git Merge

Implementation in pytorch of the weight matching algorithm presented in 
[Git Re-Basin: Merging Models modulo Permutation Symmetries"](https://arxiv.org/abs/2209.04836)

#  description
This project is part of my final exam for the Neural Network 2023 class from Sapienza university. 

This repository contains a main notebook, some models weight and a some object. 
The use of the object will be explain in the main notebook [GIT_MERGE](https://github.com/gg-Dema/NN_exam_project/blob/main/GIT_MERGE.ipynb).
The notebook contains also a small report and some personal consideration. 

The solution implemented by my are original (no reference to the original code base) and fully working over the main problem 




## Usage


The notebook contains 2 main section: 
- the definition of some function
- the experiment part

More in details, the first part contains the following section: 
- import the lib [import lib]
- download the datasets [Dataset and Dataloader]
- create the Network class [Models definiton & utility]
- define all the function for the permutation part [Permutation utility]


After this, it's possible train any desired MLP (as show in the experiment part) and merge with another MLP with the same architecture. 
I think about 2 principal way for use the permutation utility: 

Evaluation of the compatibility between 2 models: (obtain explicit the permutation)
```python 
model_A, model_B = Net(), SNet()
model_A, model_B = model_A.to(device), model_B.to(device)

loss_fn = nn.CrossEntropyLoss()

#train model A
#train model B

params_A, params_B = dict(model_A.named_parameters()), dict(model_B.named_parameters())
params_A, params_B = detach_parameters(params_A), detach_parameters(params_B)

P = get_empty_P_as_dict(params_B)
A_funct = get_A_functs(params_A, params_B)
info_perm = permutation_model_info(params_B)

new_P = weight_matching(A_funct, P)
```
Simple interpolation of 2 models: (git merge, obtain directly a new model)

```python 
model_a = Net() 
# train model_a
model_b = Net()
# train model_b
permuted_model = git_merge(model_a, model_b, silent=True)

```

for evaluate the possible permutation use: 
```python 
coef_vector = np.linspace(start=0.0, stop=1.0, num=10)
eval_dict_non_permuted = evaluate_interpolation(data_loaders['choosen_dataset'],
                                            loss_fn,
                                            model_A
                                            model_B, 
                                            coef_vector, device=device)

eval_dict_permuted = evaluate_interpolation(data_loaders['choosen_dataset'],
                                            loss_fn,
                                            model_A
                                            permuted_B, 
                                            coef_vector, device=device)
plot_barrier(eval_dict_non_permuted, eval_dict_permuted, coef_vector, 'title')

```



There is also a part for the multiple model merging, but i was unable to reproduce the performance of the paper over this part 
