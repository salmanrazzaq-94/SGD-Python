# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:56:17 2020

@author: Muhammad Salman Razzaq


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
from IPython.lib.latextools import latex_to_png

""" Displaying the equations used for optimization """

parameter_optimization = r'\theta^{+} = \theta^{-} - \frac{\alpha}{m} (h(x_{i}) - y_{i} )\bar{x}'
cost_function = r'J(x, \theta, y) = \frac{1}{2m}\sum_{i=1}^{m}(h(x_i) - y_i)^2'
estimated_point = r'h(x_i) = \theta^T \bar{x}'
print('In linear regression, I derived the equation to update the linear model parameters as:')
display(Image(data=latex_to_png(parameter_optimization, wrap=True)))
print('This minimizes the following cost function:')
display(Image(data=latex_to_png(cost_function, wrap=True)))
print('where')
display(Image(data=latex_to_png(estimated_point, wrap=True)))


""" Generate Random Data for testing the program """

slope_true = 12.963
intercept_true = 4.525
input_variable = np.arange(0.0,200.0)
output_variable = slope_true * input_variable + intercept_true + 500 * np.random.rand(len(input_variable)) - 250


""" Using Matplotlib to plot and visualize the data """
plt.figure()
plt.scatter(input_variable,output_variable)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Define Cost Function
def cost_function(input_variable, output_variable, parameters):
    "Compute Linear Regression cost"
    number_of_samples = len(input_variable)
    cost_sum = 0.0
    for x,y in zip(input_variable, output_variable):
        y_hat = np.dot(parameters, np.array([1.0, x]))
        cost_sum += (y_hat - y) ** 2 
    cost = cost_sum / (number_of_samples * 2.0)
    return cost

#Define Linear Regression Batch Gradient Descent
def linear_regression_batch_gradient_descent(input_variable, output_variable, 
                                              parameters, alpha, max_iterations):
    "Compute the params for linear regression using batch gradient descent" 
    iteration = 0
    number_of_samples = len(input_variable)
    cost = np.zeros(max_iterations)
    parameters_store = np.zeros([2, max_iterations])
    total_iterations_parameters = 0
    
    while (iteration < max_iterations):
        cost[iteration] = cost_function(input_variable, output_variable, parameters)
        parameters_store[:,iteration] = parameters
        
        #print('-----------------------------')
        #print('Iteration: {}' .format(iteration))
        #print('Cost b : {}' .format(cost[iteration]))
        
        for x,y in zip(input_variable, output_variable):
            y_hat = (np.dot(parameters, np.array([1.0,x])))
            gradient = np.array([1.0,x])*(y_hat-y)/(number_of_samples)
            parameters -= gradient * alpha
            total_iterations_parameters += 1

            
            
        iteration += 1
    
    return parameters, cost, parameters_store, total_iterations_parameters



def linear_regression_stochastic_gradient_descent(input_variable, output_variable, 
                                                  parameters, alpha, max_iterations, batch_size=1):
    """Compute the parameters for linear regression using generalized stochastic gradient descent"""
    num_samples = len(input_variable)
    cost = np.zeros(max_iterations)
    parameters_store = np.zeros([2, max_iterations])
    total_iterations_parameters = 0
    iteration = 0
    
    while iteration < max_iterations:
        list_of_samples = np.random.choice(num_samples, batch_size)
        parameters_store[:, iteration] = parameters
        cost[iteration] = cost_function(input_variable, output_variable, parameters)
        #print('-----------------------------')
        #print('Iteration: {}' .format(iteration))
        #print('Cost: {}' .format(cost[iteration]))
        #print('sample nos.:{}' .format(list_of_samples))
        sample_num = 0
        for sample in list_of_samples:
            y_hat = np.dot(parameters, np.array([1.0, input_variable[sample]]))
            gradient = np.array([1.0, input_variable[sample]]) * (output_variable[sample] - y_hat)
            parameters += alpha * gradient/num_samples
            total_iterations_parameters += 1
            sample_num +=1
            #print(f'parameters c2: {parameters}')
        
        iteration += 1
            
    return parameters, cost, parameters_store, total_iterations_parameters


"""Splitting the dataset"""
x_train, x_test, y_train, y_test = train_test_split(input_variable, output_variable, test_size=0.20)


"Training the model using batch gradient descent after initializing all the required parameters" 
initial_parameters_batch = np.array([30.0,90.0])
alpha_batch = 1e-3
max_iterations = 100
parameters_hat_batch,  cost_batch, parameters_store_batch, total_iterations_parameters_batch = linear_regression_batch_gradient_descent(x_train, y_train, initial_parameters_batch, alpha_batch, max_iterations)

                        
"Training the model using stochastic gradient descent after initializing all the required parameters"
alpha_stochastic = 1e-3
initial_parameters_stochastic = np.array([20.0, 90.0])
max_iterations2 = 100
batch_size=1
parameters_hat_stochastic,  cost_stochastic, parameters_store_stochastic, total_iterations_parameters_stochastic =linear_regression_stochastic_gradient_descent(x_train, y_train, initial_parameters_stochastic, alpha_stochastic, max_iterations2, batch_size= batch_size)

"Training the model using stochastic gradient descent after initializing all the required parameters"
alpha_minibatch = 1e-3
initial_parameters_minibatch = np.array([20.0, 90.0])
max_iterations3 = 100
batch_size2=20
parameters_hat_minibatch,  cost_minibatch, parameters_store_minibatch, total_iterations_parameters_minibatch =linear_regression_stochastic_gradient_descent(x_train, y_train, initial_parameters_minibatch, alpha_minibatch, max_iterations3, batch_size= batch_size2)


"Plotting the best parameter lines on train set as per the batch gradient descent and stochastic gradient descent methods"
plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, parameters_hat_batch[0] + parameters_hat_batch[1]*x_train, 'g', label='batch')
plt.plot(x_train, parameters_hat_stochastic[0] + parameters_hat_stochastic[1]*x_train, '-r', label='stochastic')
plt.plot(x_train, parameters_hat_minibatch[0] + parameters_hat_minibatch[1]*x_train, 'b', label='stochastic')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

"Plotting the best parameter lines on test set as per the batch gradient descent and stochastic gradient descent methods"
plt.figure()
plt.scatter(x_test, y_test)
plt.plot(x_test, parameters_hat_batch[0] + parameters_hat_batch[1]*x_test, 'g', label='batch')
plt.plot(x_test, parameters_hat_stochastic[0] + parameters_hat_stochastic[1]*x_test, '-r', label='stochastic')
plt.plot(x_test, parameters_hat_minibatch[0] + parameters_hat_minibatch[1]*x_test, 'b', label='batch')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


"Displaying best parameters and root mean square error for evaluating both methods"
print('Total Number of Iterations for finding the best parameters for Batch Gradient Descent: {}' .format(total_iterations_parameters_batch))
print('Batch best parameters => Intercept, Slope: {0:.3f} {1:.3f}' .format(parameters_hat_batch[0], parameters_hat_batch[1]))
print('Total Number of Iterations for finding the best parameters for Stochastic Gradient Descent: {}' .format(total_iterations_parameters_stochastic))
print('batch best parameters => Intercept, Slope: {0:.3f} {1:.3f}' .format(parameters_hat_stochastic[0], parameters_hat_stochastic[1]))
print('Total Number of Iterations for finding the best parameters for Batch Gradient Descent: {} with batch size {}' .format(total_iterations_parameters_minibatch, batch_size2))
print('Batch best parameters => Intercept, Slope: {0:.3f} {1:.3f}' .format(parameters_hat_minibatch[0], parameters_hat_minibatch[1]))
rms_batch = np.sqrt(np.mean(np.square(parameters_hat_batch[0] + parameters_hat_batch[1]*x_test - y_test)))
rms_minibatch = np.sqrt(np.mean(np.square(parameters_hat_minibatch[0] + parameters_hat_minibatch[1]*x_test - y_test)))
rms_stochastic = np.sqrt(np.mean(np.square(parameters_hat_stochastic[0] + parameters_hat_stochastic[1]*x_test - y_test)))
print('batch rms: {:.3f}'  .format(rms_batch))
print('batch rms: {:.3f}'  .format(rms_minibatch))
print('stochastic rms: {:.3f}' .format(rms_stochastic))

"Plotting the normalized cost against no. of iterations to estimate performance of the system"
plt.figure()
plt.plot(np.arange(max_iterations), cost_batch, 'r', label='batch')
plt.plot(np.arange(len(cost_stochastic)),  cost_stochastic, 'g', label='stochastic')
plt.plot(np.arange(len(cost_minibatch)),  cost_minibatch, 'b', label='minibatch')
plt.xlabel('iteration')
plt.ylabel('normalized cost')
plt.legend()
plt.yscale('log')
plt.show()
print('min cost with BGD: {:.3f}' .format(np.min(cost_batch)))
print('min cost with SGD: {:.3f}' .format(np.min(cost_stochastic)))
print('min cost with MBGD: {:.3f}' .format(np.min(cost_minibatch)))







