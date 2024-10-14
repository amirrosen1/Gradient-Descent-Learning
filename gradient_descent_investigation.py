import os

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR

from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go

from cross_validate import cross_validate
from loss_functions import misclassification_error

if not os.path.exists("Plots"):
    os.makedirs("Plots")


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights, norms = [], [], []

    def callback(val, weights_, **kwargs):
        values.append(val)
        weights.append(weights_.copy())
        norms.append(np.linalg.norm(weights_))

    return callback, values, weights, norms


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = {"L1": L1, "L2": L2}

    for module_name, module_cls in modules.items():
        results = {}

        for eta in etas:
            callback, vals, weights, norms = get_gd_state_recorder_callback()

            # Initialize and fit the gradient descent with the fixed learning rate
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(module_cls(weights=np.copy(init)), None, None)

            print(f"Final Loss for {module_name} with eta={eta}: {vals[-1]}")

            results[eta] = (vals, weights, norms)

            descent_path = np.array([init] + weights)
            fig = plot_descent_path(module_cls, descent_path, title=f"{module_name} - Learning Rate: {eta}")
            fig.write_html(f"Plots/gd_{module_name}_eta_{eta}.html")

        norm_fig = go.Figure(layout=go.Layout(xaxis=dict(title="GD Iteration"),
                                              yaxis=dict(title="Norm of Weights"),
                                              title=f"{module_name} Weight Norm Convergence For Different Learning Rates"))

        for eta, (_, _, norms) in results.items():
            norm_fig.add_trace(go.Scatter(x=list(range(len(norms))), y=norms, mode="lines", name=f"eta={eta}"))

        norm_fig.write_html(f"Plots/gd_{module_name}_weight_norm_convergence.html")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    fig = go.Figure(
        [go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash', width=2), showlegend=False),
         go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color="blue", width=3), text=thresholds, showlegend=False,
                    hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"{title} - AUC={auc_score:.3f}",
                         width=600, height=600, xaxis=dict(title="FPR", showgrid=True),
                         yaxis=dict(title="TPR", showgrid=True)))
    fig.write_html(f"Plots/{title.replace(' ', '_').lower()}.html")
    return thresholds, fpr, tpr


def find_optimal_threshold(tpr, fpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def run_cross_validation(X_train, y_train, penalty, lambda_values, learning_rate, max_iterations):
    scores = np.zeros((len(lambda_values), 2))
    for i, lam in enumerate(lambda_values):
        gradient_descent = GradientDescent(learning_rate=FixedLR(learning_rate), max_iter=max_iterations)
        logistic_model = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=lam, alpha=0.5)
        train_scores, val_scores = cross_validate(estimator=logistic_model, X=X_train, y=y_train,
                                                  scoring=misclassification_error)
        scores[i, 0] = np.mean(train_scores)
        scores[i, 1] = np.mean(val_scores)
        print(
            f'Penalty: {penalty}, Lambda: {lam}, Train error: {scores[i, 0]:.4f}, Validation error: {scores[i, 1]:.4f}')
    return scores


def plot_cross_validation_results(lambda_values, scores, penalty):
    fig = go.Figure([go.Scatter(x=lambda_values, y=scores[:, 0], name="Train Error"),
                     go.Scatter(x=lambda_values, y=scores[:, 1], name="Validation Error")],
                    layout=go.Layout(title=f"Train and Validation errors for {penalty} penalty",
                                     xaxis=dict(title="Lambda", type="log"),
                                     yaxis=dict(title="Error Value")))
    fig.write_html(f"Plots/{penalty}_logistic_cross_validation_errors.html")


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Fitting logistic regression model and using ROC curve for specifying threshold
    callback, losses, weights, norms = get_gd_state_recorder_callback()
    learning_rate = 1e-4
    max_iterations = 20000
    gradient_descent = GradientDescent(learning_rate=FixedLR(learning_rate), max_iter=max_iterations, callback=callback)
    logistic_model = LogisticRegression(solver=gradient_descent).fit(X_train.values, y_train.values)

    print("Final training loss:", losses[-1])
    predicted_probabilities = logistic_model.predict_proba(X_train.values)

    thresholds, fpr, tpr = plot_roc_curve(y_train, predicted_probabilities, "ROC Curve - Logistic Regression")

    # Model test error for threshold maximizing TPR-FPR
    optimal_threshold = find_optimal_threshold(tpr, fpr, thresholds)
    logistic_model.alpha_ = optimal_threshold
    test_error = logistic_model._loss(X_test.values, y_test.values)

    print(f"Optimal threshold: {optimal_threshold}\nTest error at optimal threshold: {test_error}")

    # Fitting l1-regularized logistic regression model, using cross-validation to specify values of regularization parameter
    lambda_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    penalty = "l1"

    scores = run_cross_validation(X_train.values, y_train.values, penalty, lambda_values, learning_rate,
                                  max_iterations)
    plot_cross_validation_results(lambda_values, scores, penalty)

    optimal_lambda = lambda_values[np.argmin(scores[:, 1])]
    gradient_descent = GradientDescent(learning_rate=FixedLR(learning_rate), max_iter=max_iterations)
    logistic_model = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=optimal_lambda, alpha=0.5)
    logistic_model.fit(X_train.values, y_train.values)
    test_error = logistic_model._loss(X_test.values, y_test.values)

    print(f"Penalty: {penalty}\nOptimal Lambda: {optimal_lambda}\nModel test error: {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
