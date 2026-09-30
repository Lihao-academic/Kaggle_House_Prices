# 本文件存放计算BIC的相关函数和类
# 包括BIC计算公式和transformer兼容的类

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_BIC(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    MSE = mean_squared_error(y_train, y_pred)
    nrows = X_train.shape[0]
    ncols = X_train.shape[1]
    RSS = MSE * nrows
    return nrows * np.log(RSS / nrows) + (ncols + 1) * np.log(nrows)


class BICSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X_train, y_train):

        # 这里运行backward BIC
        # 先获得初始BIC
        current_BIC = calculate_BIC(X_train, y_train)
        # 初始列名，维护一个当前存在的列
        all_cols_name = X_train.columns.tolist()
        # selected_columns = X_train.columns.tolist()

        X_train_copy = X_train.copy()
        # 准备变量
        cols_to_drop = None
        history_bic = []
        step = 1
        # 进行循环，直到最后选出所有的列名
        print("进入循环，开始测试各列并计算分数")
        start_time = time.time()
        while True:
            # 首先如果只剩一列，直接跳出
            if X_train_copy.shape[1] <= 1:
                break
            #
            best_BIC = current_BIC

            # 不需要flag，改成cols_to_drop
            cols_to_drop = None
            # 在当前的所有列里面循环
            for cols in all_cols_name:
                # 逐个删除列，逐个计算BIC
                new_BIC = calculate_BIC(X_train_copy.drop(cols, axis=1), y_train)
                # 如果刚刚删除一个列得到的BIC更小，就记下来，并更新当前最小的BIC
                if new_BIC < best_BIC:
                    cols_to_drop = cols
                    best_BIC = new_BIC
            # 如果一个循环结束，没有发现任何新的可删除列
            # 就说明已经穷尽了，直接跳出while
            if cols_to_drop is None:
                break

            # 更新阶段，把该删的删掉，更新“当前所有列”，记录历史
            # 删除本轮确定的列，更新当前最低BIC，更新当前列名
            end_time = time.time()
            print(f"已经完成一次删列，当前已删除列数: {step}, 当前BIC: {round(best_BIC, 2)}, 当前耗时 {int(end_time - start_time)} 秒.")
            X_train_copy = X_train_copy.drop(cols_to_drop, axis=1)
            current_BIC = best_BIC
            all_cols_name = X_train_copy.columns.tolist()
            history_bic.append((step, cols_to_drop, current_BIC))
            step +=1

        self.selected_columns_ = X_train_copy.columns.tolist()
        self.final_bic_ = current_BIC
        self.history_bic_ = history_bic

        return self
    def transform(self, X_train):
        return X_train[self.selected_columns_]