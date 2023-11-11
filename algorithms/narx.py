from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from collections import deque
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from utils import shift, MetaLagFeatureProcessor
from sklearn.model_selection import GridSearchCV



class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    """
    TimeSeriesRegressor creates a time series model based on a
    general-purpose regression model defined in base_estimator.
    base_estimator must be a model which implements the scikit-learn APIs.
    """

    def __init__(self, base_estimator, **base_params):
        self.base_estimator = base_estimator.set_params(**base_params)

    def set_params(self, **params):
        for param, value in params.items():
            if param in self.get_params():
                super(TimeSeriesRegressor, self).set_params(**{param: value})
            else:
                self.base_estimator.set_params(**{param: value})
        return self


class GeneralAutoRegressor(TimeSeriesRegressor, RegressorMixin):
    r"""
    The general auto regression model can be written in the following form:

    .. math::
        y(t + k) &=& f(y(t), ..., y(t-p+1), \\
                 & & x_1(t - d_1), ..., x_1(t-d_1-q_1+1), \\
                 & & ..., x_m(t - d_1), ..., x_m(t - d_m - q_m + 1)) + e(t)
        :label: gar

    :param object base_estimator: an estimator object that implements the
                                  scikit-learn API (fit, and predict). The
                                  estimator will be used to fit the function
                                  :math:`f` in equation :eq:`gar`.
    :param int auto_order: the autoregression order :math:`p` in equation
                           :eq:`gar`.
    :param list exog_order: the exogenous input order, a list of integers
                            representing the order for each exogenous input,
                            i.e. :math:`[q_1, q_2, ..., q_m]` in equation
                            :eq:`gar`.
    :param list exog_delay: the delays of the exogenous inputs, a list of
                            integers representing the delay of each exogenous
                            input, i.e. :math:`[d_1, d_2, ..., d_m]` in
                            equation :eq:`gar`. By default, all the delays are
                            set to 0.
    :param int pred_step: the prediction step :math:`k` in equation :eq:`gar`.
                          By default, it is set to 1.
    :param dict base_params: other keyword arguments for base_estimator.
    """

    def __init__(self,
                 base_estimator,
                 auto_order,
                 exog_order,
                 exog_delay=None,
                 pred_step=1,
                 **base_params):
        super(GeneralAutoRegressor, self).__init__(base_estimator,
                                                   **base_params)
        self.auto_order = auto_order
        self.exog_order = exog_order
        if exog_delay is None:
            exog_delay = [0] * len(exog_order)
        if len(exog_delay) != len(exog_order):
            raise ValueError(
                'The length of exog_delay must be the same as the length of exog_order.'
            )
        self.exog_delay = exog_delay
        self.num_exog_inputs = len(exog_order)
        self.pred_step = pred_step

    def fit(self, X, y, **params):
        """
        Create lag features and fit the base_estimator.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        """
        X, y = self._check_and_preprocess_X_y(X, y)
        features, target = self._preprocess_data(X, y)
        self.base_estimator.fit(features, target, **params)

    def _preprocess_data(self, X, y):
        """
        Helper function to prepare the data for base_estimator.
        """
        p = self._get_lag_feature_processor(X, y)
        features = p.generate_lag_features()
        target = shift(y, -self.pred_step)

        # Remove NaN introduced by shift
        all_data = np.concatenate([target.reshape(-1, 1), features], axis=1)
        mask = np.isnan(all_data).any(axis=1)
        features, target = features[~mask], target[~mask]
        return features, target

    def _get_lag_feature_processor(self, X, y):
        return MetaLagFeatureProcessor(X, y, self.auto_order, self.exog_order,
                                       self.exog_delay)

    def grid_search(self, X, y, para_grid, **params):
        """
        Perform grid search on the base_estimator. The function first generates
        the lag features and predicting targets, and then calls
        ``GridSearchCV`` in scikit-learn package.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param dict para_grid: use the same format in ``GridSearchCV`` in
                               scikit-learn package.
        :param dict params: other keyword arguments that can be passed into
                            ``GridSearchCV`` in scikit-learn package.
        """
        grid = GridSearchCV(self.base_estimator, para_grid, **params)
        X, y = self._check_and_preprocess_X_y(X, y)
        features, target = self._preprocess_data(X, y)
        grid.fit(features, target)
        self.set_params(**grid.best_params_)

    def _predictNA(self, Xdata):
        # Xdata contains nan introduced by shift
        ypred = np.empty(Xdata.shape[0]) * np.nan
        mask = np.isnan(Xdata).any(axis=1)
        X2pred = Xdata[~mask]
        ypred[~mask] = self.base_estimator.predict(X2pred)
        return ypred

    def _check_and_preprocess_X_y(self, X, y):
        min_samples_required = max(self.auto_order,
                np.max(np.array(self.exog_delay) + np.array(self.exog_order))) - 1
        X, y = check_X_y(X, y, ensure_min_samples=min_samples_required)
        if len(self.exog_order) != X.shape[1]:
            raise ValueError(
                'The number of columns of X must be the same as the length of exog_order.'
            )
        return X, y



class NARX(GeneralAutoRegressor):
    r"""
    NARX stands for `Nonlinear AutoRegressive eXogenous model
    <https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model>`_.
    The model equation is written as follows.

    .. math::
        y(t + 1) &=& f(y(t), ..., y(t-p+1), \\
                 & & x_1(t - d_1), ..., x_1(t-d_1-q_1+1), \\
                 & & ..., x_m(t - d_1), ..., x_m(t - d_m - q_m + 1)) + e(t)
        :label: narx

    :param object base_estimator: an estimator object that implements the
                                  scikit-learn API (fit, and predict). The
                                  estimator will be used to fit the function
                                  :math:`f` in equation :eq:`narx`.
    :param int auto_order: the autoregression order :math:`p` in equation
                           :eq:`narx`.
    :param list exog_order: the exogenous input order, a list of integers
                            representing the order for each exogenous input,
                            i.e. :math:`[q_1, q_2, ..., q_m]` in equation
                            :eq:`narx`.
    :param list exog_delay: the delays of the exogenous inputs, a list of
                            integers representing the delay of each exogenous
                            input, i.e. :math:`[d_1, d_2, ..., d_m]` in
                            equation :eq:`narx`. By default, all the delays are
                            set to 0.
    :param dict base_params: other keyword arguments for base_estimator.
    """

    def __init__(self,
                 base_estimator,
                 auto_order,
                 exog_order,
                 exog_delay=None,
                 **base_params):
        super(NARX, self).__init__(
            base_estimator,
            auto_order,
            exog_order,
            exog_delay=exog_delay,
            pred_step=1,
            **base_params)

    def score(self, X, y, step=1, method="r2"):
        """
        Produce multi-step prediction of y, and compute the metrics against y.
        Nan is ignored when computing the metrics.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param int step: prediction step.
        :param string method: could be "r2" (R Square) or "mse" (Mean Square
                              Error).

        :return: prediction metric. Nan is ignored when computing the metrics.
        """
        ypred = self.predict(X, y, step=step)
        mask = np.isnan(y) | np.isnan(ypred)
        if method == "r2":
            return r2_score(y[~mask], ypred[~mask])
        elif method == "mse":
            return mean_squared_error(y[~mask], ypred[~mask])

    def predict(self, X, y, step=1):
        r"""
        Produce multi-step prediction of y. The multi-step prediction is done
        recursively by using the future inputs in X. The prediction equation is
        as follows:

        .. math::
            \hat{y}(t + k) &=& f(\hat{y}(t + k - 1), ..., \hat{y}(t + k - p), \\
                           & &x_1(t + k - 1 - d_1), ..., x_1(t + k - d_1 - q_1) \\
                           & &..., x_m(t + k - 1 - d_m), ..., x_m(t + k - d_m - q_m))

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param int step: prediction step.

        :return: k-step prediction time series, shape = (n_samples). The
                 :math:`i` th value of the output is the k-step prediction of
                 the :math:`i` th value of the input ``y``. The first ``step +
                 max(auto_order - 1, max(exog_order + exog_delay) - 1)`` values of the
                 output is ``np.nan``.
        """
        X, y = self._check_and_preprocess_X_y(X, y)
        p = self._get_lag_feature_processor(X, y)
        features = p.generate_lag_features()

        for k in range(step):
            yhat = self._predictNA(features)
            if k == step - 1:
                break
            features = p.update(yhat)

        ypred = np.concatenate([np.empty(step) * np.nan, yhat])[0:len(y)]
        return ypred

    def forecast(self, X, y, step=1, X_future=None):
        r"""
        Forecast y multiple step ahead given the exogenous input history X, 
        output history y and the future exogenous input X_future. X_future is
        assumed to be all zeros if not specified.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param int step: prediction step.
        :param array-like X_futrue: future exogenous input time series, shape =
                                    (step - 1, n_exog_inputs)

        :return: multi-step forecasted time series, shape = (step).
        """
        assert step > 0

        X, y = self._check_and_preprocess_X_y(X, y)

        if X_future is None:
            X_future = np.zeros((step - 1, self.num_exog_inputs))
        X_future = check_array(X_future, ensure_min_samples=0)
        if X_future.shape[0] != step - 1:
            raise ValueError('The row number of X_future ({}) must to step - 1 ({})!'.format(X_future.shape[0], step - 1))

        auto_regressor = deque(y[:(-1 - self.auto_order):-1])
        exog_regressors = [
                deque(X[(-1 - d):(-1 - d - q):-1, i])
                for i, (d, q) in enumerate(zip(self.exog_delay, self.exog_order))
                ]
        cur_step = 0
        y_forecast = []
        while cur_step < step:
            X_base = np.concatenate([np.array(auto_regressor), 
                np.concatenate(exog_regressors)]).reshape(1, -1)
            y_hat = self.base_estimator.predict(X_base)
            y_forecast.append(y_hat[0])
            if cur_step == step - 1:
                break
            # update regressors with the newly obtained values
            auto_regressor.pop()
            auto_regressor.appendleft(y_forecast[-1])
            for exog_reg, X_next in zip(exog_regressors, X_future[cur_step, :]):
                exog_reg.pop()
                exog_reg.appendleft(X_next)
            cur_step += 1
        return np.array(y_forecast)


class DirectAutoRegressor(GeneralAutoRegressor):
    r"""
    This model performs autoregression with exogenous inputs on the k-step
    ahead output directly. The model equation is written as follows.

    .. math::
        y(t + k) &=& f(y(t), ..., y(t-p+1), \\
                 & & x_1(t - d_1), ..., x_1(t-d_1-q_1+1), \\
                 & & ..., x_m(t - d_1), ..., x_m(t - d_m - q_m + 1)) + e(t)
        :label: direct

    :param object base_estimator: an estimator object that implements the
                                  scikit-learn API (fit, and predict). The
                                  estimator will be used to fit the function
                                  :math:`f` in equation :eq:`direct`.
    :param int auto_order: the autoregression order :math:`p` in equation
                           :eq:`direct`.
    :param list exog_order: the exogenous input order, a list of integers
                            representing the order for each exogenous input,
                            i.e. :math:`[q_1, q_2, ..., q_m]` in equation
                            :eq:`direct`.
    :param int pred_step: the prediction step :math:`k` in equation :eq:`gar`.
                          By default, it is set to 1.
    :param list exog_delay: the delays of the exogenous inputs, a list of
                            integers representing the delay of each exogenous
                            input, i.e. :math:`[d_1, d_2, ..., d_m]` in
                            equation :eq:`direct`. By default, all the delays
                            are set to 0.
    :param dict base_params: other keyword arguments for base_estimator.
    """

    def __init__(self,
                 base_estimator,
                 auto_order,
                 exog_order,
                 pred_step,
                 exog_delay=None,
                 **base_params):
        super(DirectAutoRegressor, self).__init__(
            base_estimator,
            auto_order,
            exog_order,
            exog_delay=exog_delay,
            pred_step=pred_step,
            **base_params)

    def predict(self, X, y):
        r"""
        Produce multi-step prediction of y. The multi-step prediction is done
        directly. No future X inputs are used in the prediction. The prediction
        equation is as follows:

        .. math::
            \hat{y}(t + k) &=&  f(y(t), ..., y(t - p + 1), \\
                           & & x_1(t - d_1), ..., x_1(t - d_1 - q_1 + 1) \\
                           & & ..., x_m(t - d_m), ..., x_m(t - d_m - q_m + 1))

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param int step: prediction step.

        :return: k-step prediction time series, shape = (n_samples). The
                 :math:`i` th value of the output is the k-step prediction of
                 the :math:`i` th value of the input ``y``. The first
                 ``pred_step + max(auto_order - 1, max(exog_order +
                 exog_delay) - 1)`` values of the output is ``np.nan``.
        """
        X, y = self._check_and_preprocess_X_y(X, y)
        p = self._get_lag_feature_processor(X, y)
        features = p.generate_lag_features()
        yhat = self._predictNA(features)

        ypred = np.concatenate([np.empty(self.pred_step) * np.nan,
                                yhat])[0:len(y)]
        return ypred

    def score(self, X, y, method="r2", verbose=False):
        """
        Produce multi-step prediction of y, and compute the metrics against y.
        Nan is ignored when computing the metrics.

        :param array-like X: exogenous input time series, shape = (n_samples,
                             n_exog_inputs)
        :param array-like y: target time series to predict, shape = (n_samples)
        :param string method: could be "r2" (R Square) or "mse" (Mean Square
                              Error).

        :return: prediction metric. Nan is ignored when computing the metrics.
        """
        ypred = self.predict(X, y)
        mask = np.isnan(y) | np.isnan(ypred)
        if verbose:
            print('Evaluating {} score, {} of {} data points are evaluated.'.
                  format(method, np.sum(~mask), y.shape[0]))
        if method == "r2":
            return r2_score(y[~mask], ypred[~mask])
        elif method == "mse":
            return mean_squared_error(y[~mask], ypred[~mask])
        else:
            raise ValueError('{} method is not supported. Please choose from \"r2\" or \"mse\".')